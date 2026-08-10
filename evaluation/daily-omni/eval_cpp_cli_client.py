"""
llama-omni-eval-daily-cli 子进程客户端（Daily-Omni 版）。

替代原先的 HTTP `llama-server` 方案：每个 GPU 启动一个常驻 CLI 进程，模型只加载
一次，之后通过 stdin/stdout 管道用 JSONL 协议逐题推理，避免 HTTP/SSE 的网络与
序列化开销。

与 Video-MME 版的区别：infer 请求多带一个 `audios` 数组，CLI 侧会按
[frame_0, audio_0, frame_1, audio_1, ...] 交错 prefill（对齐 evalkit 的
content = [PIL.Image, np.ndarray, ..., str] 结构）。

协议（每行一个 JSON）：
  发送 (Python -> CLI, stdin):
    {"type":"infer","id":"vid-1","frames":[...],"audios":[...],"prompt":"...",
     "max_slice_nums":0,"n_predict":128}
    {"type":"ping"} / {"type":"quit"}
  接收 (CLI -> Python, stdout，已与模型日志分离，是干净的 JSONL):
    {"type":"ready"} / {"type":"pong"}
    {"type":"result","id":"vid-1","ok":true,"response":"A"}
    {"type":"result","id":"vid-1","ok":false,"error":"..."}

CLI 侧把模型/ggml 的 stdout 噪声重定向到 stderr（落到日志文件），因此本客户端从
子进程 stdout 读到的每一行都是协议 JSON。
"""
import os
import json
import time
import signal
import select
import logging
import subprocess
from glob import glob
from typing import List, Optional

from eval_cpp_config import (
    LLAMA_CLI_BIN, LLM_MODEL_PATH, CTX_SIZE,
    MAX_SLICE_NUMS, MAX_TOKENS,
    TEMPERATURE, TOP_P, TOP_K, REPEAT_PENALTY, SAMPLER_SEED,
    CLI_STARTUP_TIMEOUT, INFER_TIMEOUT, MAX_CLI_RESTARTS,
)

logger = logging.getLogger(__name__)


class CliUnhealthy(RuntimeError):
    """CLI 卡死或已退出。重启能恢复，单题的模型错误不属于这一类。"""


class CliUnrecoverable(RuntimeError):
    """重启次数用尽仍不可用。整个任务应当失败退出，而不是把余下题目都记成答错。"""


def _rotate_logs(log_dir: str, gpu_id: int, keep: int = 5) -> str:
    """轮转 CLI 日志，避免单文件无限变大。当前运行写 cli_gpu{gpu_id}.log。"""
    os.makedirs(log_dir, exist_ok=True)
    active = os.path.join(log_dir, f"cli_gpu{gpu_id}.log")
    if os.path.exists(active) and os.path.getsize(active) > 0:
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.replace(active, os.path.join(log_dir, f"cli_gpu{gpu_id}_{ts}.log"))
    rotated = sorted(glob(os.path.join(log_dir, f"cli_gpu{gpu_id}_*.log")),
                     key=os.path.getmtime, reverse=True)
    for old in rotated[keep:]:
        try:
            os.remove(old)
        except OSError:
            pass
    return active


class OmniCliClient:
    """封装单个 llama-omni-eval-daily-cli 常驻进程。"""

    def __init__(
        self,
        gpu_id: int,
        model_path: str = LLM_MODEL_PATH,
        ctx_size: int = CTX_SIZE,
        max_slice_nums: int = MAX_SLICE_NUMS,
        n_predict: int = MAX_TOKENS,
        log_dir: Optional[str] = None,
        keep_rotated_logs: int = 5,
    ):
        self.gpu_id = gpu_id
        self.max_slice_nums = max_slice_nums
        self.n_predict = n_predict

        if not os.path.isfile(LLAMA_CLI_BIN):
            raise FileNotFoundError(f"eval CLI binary not found: {LLAMA_CLI_BIN}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        env = os.environ.copy()
        # 设备变量名随后端变：CUDA 是 CUDA_VISIBLE_DEVICES，Ascend/CANN 是
        # ASCEND_RT_VISIBLE_DEVICES。父进程给的是可用卡列表（如 4,5,6,7），
        # 这里取第 gpu_id 个物理卡，让每个 worker 只看见自己那张。
        dev_var = os.environ.get("DEVICE_ENV_VAR", "CUDA_VISIBLE_DEVICES")
        parent_visible = os.environ.get(dev_var, "")
        if parent_visible:
            devices = [d.strip() for d in parent_visible.split(",") if d.strip()]
            env[dev_var] = devices[gpu_id] if gpu_id < len(devices) else str(gpu_id)
        else:
            env[dev_var] = str(gpu_id)

        extra_ld = os.environ.get("EXTRA_LD_LIBRARY_PATH", "")
        if extra_ld:
            ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{extra_ld}:{ld}" if ld else extra_ld

        cmd = [
            LLAMA_CLI_BIN,
            "-m", model_path,
            "-c", str(ctx_size),
            "-ngl", "999",
            "--max-slice-nums", str(max_slice_nums),
            "--n-predict", str(n_predict),
            "--temp", str(TEMPERATURE),
            "--top-p", str(TOP_P),
            "--top-k", str(TOP_K),
            "--repeat-penalty", str(REPEAT_PENALTY),
            "--seed", str(SAMPLER_SEED),
        ]

        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")

        # 重启要用同一套命令和环境，所以存下来。
        self._cmd = cmd
        self._env = env
        self._log_dir = log_dir
        self._keep_rotated_logs = keep_rotated_logs
        self.restarts = 0
        self._spawn()

    # ---------------- 进程生命周期 ----------------

    def _spawn(self) -> None:
        """起一个新的 CLI 子进程。日志会轮转，上一条命的日志留着排查。"""
        self.log_path = _rotate_logs(self._log_dir, self.gpu_id,
                                     keep=self._keep_rotated_logs)
        self._log_f = open(self.log_path, "w", encoding="utf-8", buffering=1)
        logger.info(f"[GPU {self.gpu_id}] launching daily CLI: {' '.join(self._cmd)}"
                    f"  (log: {self.log_path})")
        self.proc = subprocess.Popen(
            self._cmd, env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,   # 干净的协议通道
            stderr=self._log_f,       # 模型/ggml 噪声
            text=True, bufsize=1,
            preexec_fn=os.setsid,
        )

    def _kill(self) -> None:
        """强杀整个进程组。卡在设备调用里的进程不会响应协议层的 quit。"""
        if self.proc.poll() is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(self.proc.pid), sig)
                    self.proc.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    break
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream and not stream.closed:
                    stream.close()
            except Exception:
                pass
        try:
            if self._log_f and not self._log_f.closed:
                self._log_f.close()
        except Exception:
            pass

    def restart(self) -> bool:
        """杀掉重启并等待就绪。超过重启上限直接拒绝。"""
        if self.restarts >= MAX_CLI_RESTARTS:
            return False
        self.restarts += 1
        logger.warning(f"[GPU {self.gpu_id}] 重启 CLI"
                       f"（第 {self.restarts}/{MAX_CLI_RESTARTS} 次）")
        self._kill()
        self._spawn()
        return self.wait_ready()

    # ---------------- 底层协议 ----------------

    def _send(self, obj: dict) -> None:
        if self.proc.poll() is not None:
            raise CliUnhealthy(
                f"[GPU {self.gpu_id}] CLI process已退出 (code={self.proc.returncode})")
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _read_json(self, timeout: float) -> Optional[dict]:
        """读一行协议 JSON。超时或 EOF 返回 None（非 JSON 行会被跳过）。"""
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            rlist, _, _ = select.select([self.proc.stdout], [], [], remaining)
            if not rlist:
                return None
            line = self.proc.stdout.readline()
            if line == "":  # EOF：进程结束
                return None
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # 理论上不该出现，稳妥起见跳过非协议行
            if isinstance(obj, dict) and "type" in obj:
                return obj

    # ---------------- 生命周期 ----------------

    def wait_ready(self, timeout: float = CLI_STARTUP_TIMEOUT) -> bool:
        """等待模型加载完成（收到 {"type":"ready"}）。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                logger.error(f"[GPU {self.gpu_id}] CLI 启动即退出 (code={self.proc.returncode})，见日志 {self.log_path}")
                return False
            obj = self._read_json(timeout=min(10.0, timeout))
            if obj is None:
                continue
            if obj.get("type") == "ready":
                logger.info(f"[GPU {self.gpu_id}] CLI ready，耗时 {time.time()-t0:.1f}s")
                return True
            if obj.get("type") == "fatal":
                logger.error(f"[GPU {self.gpu_id}] CLI fatal: {obj.get('error')}")
                return False
        logger.error(f"[GPU {self.gpu_id}] 等待 ready 超时 ({timeout}s)")
        return False

    # ---------------- 推理 ----------------

    def infer(
        self,
        frame_paths: List[str],
        audio_paths: List[str],
        prompt: str,
        qid: str = "",
        max_slice_nums: Optional[int] = None,
        n_predict: Optional[int] = None,
        timeout: float = INFER_TIMEOUT,
    ) -> str:
        """单题推理，卡死或进程已退出时重启一次再重试。

        必须重启而不是重发请求：卡死的形态是 CLI 的线程阻塞在一个同步的设备调用里，
        它既不读 stdin 也不返回，协议层面发什么都没人接。不重启的话，该分片剩下的
        每道题都会空等满 INFER_TIMEOUT。

        每题在 CLI 侧都会 reset（清 KV、重建 system prompt），所以重启只损失当前这题，
        重试它即可，前面的结果不受影响。
        """
        try:
            return self._infer_once(frame_paths, audio_paths, prompt, qid,
                                    max_slice_nums, n_predict, timeout)
        except CliUnhealthy as e:
            logger.warning(f"[GPU {self.gpu_id}] {e}，尝试重启后重试 qid={qid}")
            if not self.restart():
                raise CliUnrecoverable(
                    f"[GPU {self.gpu_id}] 重启失败或已用尽 {MAX_CLI_RESTARTS} 次机会"
                    f"（最后一次失败于 qid={qid}）") from e
            try:
                return self._infer_once(frame_paths, audio_paths, prompt, qid,
                                        max_slice_nums, n_predict, timeout)
            except CliUnhealthy as e2:
                # 重启后立刻又卡死/退出，说明不是偶发，别再耗下去。
                raise CliUnrecoverable(
                    f"[GPU {self.gpu_id}] 重启后仍然不可用（qid={qid}）") from e2

    def _infer_once(
        self,
        frame_paths: List[str],
        audio_paths: List[str],
        prompt: str,
        qid: str,
        max_slice_nums: Optional[int],
        n_predict: Optional[int],
        timeout: float,
    ) -> str:
        """一次尝试：reset -> 交错 prefill(帧, 音频) -> prefill 文本 -> decode。"""
        req = {
            "type": "infer",
            "id": qid,
            "frames": frame_paths,
            "audios": audio_paths or [],
            "prompt": prompt,
            "max_slice_nums": self.max_slice_nums if max_slice_nums is None else max_slice_nums,
            "n_predict": self.n_predict if n_predict is None else n_predict,
        }
        self._send(req)
        obj = self._read_json(timeout=timeout)
        if obj is None:
            # 读不到东西：超时（卡死）或管道 EOF（进程没了），两者都要重启。
            raise CliUnhealthy(f"[GPU {self.gpu_id}] infer 超时或进程退出 (qid={qid})")
        if obj.get("type") != "result":
            raise RuntimeError(f"[GPU {self.gpu_id}] 非预期响应: {obj}")
        if not obj.get("ok", False):
            # CLI 还活着，只是这道题算失败了 —— 属于单题错误，不重启。
            raise RuntimeError(f"[GPU {self.gpu_id}] infer 失败 (qid={qid}): {obj.get('error')}")
        return obj.get("response", "")

    def ping(self, timeout: float = 10.0) -> bool:
        try:
            self._send({"type": "ping"})
            obj = self._read_json(timeout=timeout)
            return bool(obj) and obj.get("type") == "pong"
        except Exception:
            return False

    def close(self):
        """优雅关闭子进程。"""
        if self.proc.poll() is None:
            try:
                self._send({"type": "quit"})
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"[GPU {self.gpu_id}] CLI 未及时退出，发送 SIGTERM")
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                    self.proc.wait(timeout=5)
                except Exception:
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        if self._log_f and not self._log_f.closed:
            self._log_f.close()


def start_all_clients(
    num_gpus: int,
    model_path: str = LLM_MODEL_PATH,
    ctx_size: int = CTX_SIZE,
    keep_rotated_logs: int = 5,
) -> List[OmniCliClient]:
    """启动所有 GPU 上的 CLI 进程并等待就绪。任一失败则全部关闭并抛异常。"""
    clients: List[OmniCliClient] = []
    for gpu_id in range(num_gpus):
        clients.append(OmniCliClient(
            gpu_id, model_path=model_path, ctx_size=ctx_size,
            keep_rotated_logs=keep_rotated_logs,
        ))

    logger.info(f"Waiting for {num_gpus} CLI processes to load model...")
    failed = [c.gpu_id for c in clients if not c.wait_ready()]
    if failed:
        logger.error(f"CLI startup failed on GPUs: {failed}")
        stop_all_clients(clients)
        raise RuntimeError(f"CLI startup failed on GPUs: {failed}")

    logger.info(f"All {num_gpus} CLI processes ready")
    return clients


def stop_all_clients(clients: List[OmniCliClient]):
    for c in clients:
        try:
            c.close()
        except Exception as e:
            logger.warning(f"Error stopping CLI GPU {c.gpu_id}: {e}")
    logger.info("All CLI processes stopped")
