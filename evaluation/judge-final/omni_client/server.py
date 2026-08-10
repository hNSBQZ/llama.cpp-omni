"""llama-server 子进程启停与健康检查。"""

from __future__ import annotations

import glob
import logging
import os
import platform
import signal
import subprocess
import threading
import time
from typing import Callable, List, Optional

import sys
from pathlib import Path

_JUDGE_ROOT = Path(__file__).resolve().parents[1]
if str(_JUDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_JUDGE_ROOT))
from judge_support import display_path  # noqa: E402

logger = logging.getLogger("omni_client.server")


def sampler_seed() -> int:
    """采样种子。固定值，让同一份构建重复跑得到同一条 token 轨迹。

    默认 42，OMNI_SAMPLER_SEED 可覆盖。omni_init 请求里的 seed 也读这个函数，
    两处必须一致：--seed 决定 omni_init 建的那对 sampler，请求里的 seed 是
    server 重新 omni_init 时用的。
    """
    raw = os.environ.get("OMNI_SAMPLER_SEED", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("OMNI_SAMPLER_SEED=%r 不是整数，退回 42", raw)
    return 42


def find_server_binary(llamacpp_root: str) -> str:
    """定位 omni server 二进制。

    OMNI_SERVER_BIN 可直接指定。否则按 llama-omni-server 优先于 llama-server 搜
    build* 目录 —— 本仓库把 omni server 单独建成 llama-omni-server，构建目录也
    不一定叫 build（后端不同常另起一个）。
    """
    env_bin = os.environ.get("OMNI_SERVER_BIN")
    if env_bin:
        return env_bin

    is_win = platform.system() == "Windows"
    names = ["llama-omni-server", "llama-server"]
    if is_win:
        names = [n + ".exe" for n in names]

    build_dirs = [os.path.join(llamacpp_root, "build")]
    build_dirs += sorted(glob.glob(os.path.join(llamacpp_root, "build*")))

    candidates: List[str] = []
    for bd in build_dirs:
        for sub in ("bin", os.path.join("bin", "Release")):
            for name in names:
                p = os.path.join(bd, sub, name)
                if p not in candidates:
                    candidates.append(p)

    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


class CppServerProcess:
    """管理 llama-server 子进程生命周期。"""

    def __init__(
        self,
        llamacpp_root: str,
        model_path: str,
        port: int,
        gpu_id: int,
        ctx_size: int = 4096,
        n_gpu_layers: int = 99,
        output_dir: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> None:
        self.llamacpp_root = llamacpp_root
        self.model_path = model_path
        self.port = port
        self.gpu_id = gpu_id
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.output_dir = output_dir or os.path.join(
            llamacpp_root, f"tools/omni/output_{port}"
        )
        self.log_path = log_path
        self._process: Optional[subprocess.Popen] = None
        self.base_url = f"http://127.0.0.1:{port}"

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    def start(self) -> None:
        server_bin = find_server_binary(self.llamacpp_root)
        if not os.path.exists(server_bin):
            raise RuntimeError(f"llama-server not found: {server_bin}")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"LLM model not found: {self.model_path}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Ascend/CANN 认的是另一个变量名。设了之后进程内 device 0 就是这块物理卡，
        # omni_init 传的 token2wav_device="gpu:0" 才会落到正确的芯片上。
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(self.gpu_id)

        cmd = [
            server_bin,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--model",
            self.model_path,
            "--ctx-size",
            str(self.ctx_size),
            "--n-gpu-layers",
            str(self.n_gpu_layers),
            "--repeat-penalty",
            "1.05",
            "--temp",
            "0.7",
            "--seed",
            str(sampler_seed()),
        ]

        logger.info(
            "Starting C++ server: %s",
            " ".join(display_path(a) if isinstance(a, str) and os.path.isabs(a) else a for a in cmd),
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self._process = subprocess.Popen(
            cmd,
            env=env,
            cwd=self.llamacpp_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )

        log_path = self.log_path or os.path.join(self.output_dir, "cpp_server_stdout.log")

        def _log_reader() -> None:
            try:
                assert self._process is not None and self._process.stdout is not None
                with open(log_path, "a", encoding="utf-8") as cpp_f:
                    for line in self._process.stdout:
                        stripped = line.rstrip()
                        cpp_f.write(stripped + "\n")
                        cpp_f.flush()
            except Exception:
                pass

        threading.Thread(target=_log_reader, daemon=True).start()
        self._wait_health(timeout_s=300)

    def _wait_health(self, timeout_s: int = 300) -> None:
        import requests

        no_proxy = {"http": None, "https": None}
        for i in range(timeout_s):
            proc = self._process
            if proc is not None:
                return_code = proc.poll()
                if return_code is not None:
                    log_path = self.log_path or os.path.join(
                        self.output_dir, "cpp_server_stdout.log"
                    )
                    raise RuntimeError(
                        f"C++ server exited before becoming ready "
                        f"(code={return_code}, log={log_path})"
                    )
            try:
                r = requests.get(
                    f"{self.base_url}/health", timeout=2, proxies=no_proxy
                )
                if r.status_code == 200:
                    logger.info("C++ server ready after %ss", i + 1)
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"C++ server startup timeout ({timeout_s}s)")

    def stop(self) -> None:
        if self._process is None:
            return
        proc = self._process
        try:
            if proc.poll() is None:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except Exception:
                    proc.terminate()

                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        proc.kill()
                    proc.wait(timeout=5)
        except Exception as e:
            logger.warning("_stop_cpp_server: %s", e)
        finally:
            self._process = None
            logger.info("llama-server stopped")

    def is_alive(self) -> bool:
        proc = self._process
        if proc is None or proc.poll() is not None:
            return False
        return True
