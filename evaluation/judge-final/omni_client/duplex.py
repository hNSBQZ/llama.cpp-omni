"""DuplexSession：omni_init / prefill / decode / duplex API。"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import httpx
import numpy as np

import sys
from pathlib import Path

_JUDGE_ROOT = Path(__file__).resolve().parents[1]
if str(_JUDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_JUDGE_ROOT))
from judge_support import display_path  # noqa: E402

from . import media, prompts, wav_poller
from .server import CppServerProcess, sampler_seed

logger = logging.getLogger("omni_client.duplex")

_CPP_SAMPLING_KEYS = (
    "listen_prob_scale",
    "force_listen_count",
    "max_new_speak_tokens_per_chunk",
    "tts_temperature",
)


@dataclass
class DuplexGenerateResult:
    is_listen: bool = True
    text: str = ""
    end_of_turn: bool = False
    # 服务端三态的第三态：轮次已 turn_eos 结束、本帧零 TTS token 产出。
    # 模型没主动采样 <|listen|> 所以 is_listen 仍是 False，但语义等价于 LISTEN。
    turn_idle: bool = False
    current_time: int = 0
    cost_all_ms: float = 0.0
    cost_llm_ms: Optional[float] = None
    cost_tts_ms: Optional[float] = None
    cost_token2wav_ms: Optional[float] = None
    vpm_ms: Optional[float] = None
    apm_ms: Optional[float] = None
    llm_prefill_ms: Optional[float] = None
    stage_cnt: Optional[int] = None


@dataclass
class PrefillResult:
    n_vision_images: int = 0
    cnt: int = 0


class DuplexSession:
    """Duplex 会话驱动。"""

    def __init__(
        self,
        llamacpp_root: str,
        model_dir: str,
        llm_model: str,
        gpu_id: int = 0,
        cpp_server_port: Optional[int] = None,
        ref_audio_path: Optional[str] = None,
        ctx_size: int = 4096,
        n_gpu_layers: int = 99,
        log_path: Optional[str] = None,
    ) -> None:
        self.llamacpp_root = llamacpp_root
        self.model_dir = model_dir
        self.llm_model = llm_model
        self.gpu_id = gpu_id
        self.ref_audio_path = ref_audio_path
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers

        self._cpp_server_port = cpp_server_port or (19060 + gpu_id)
        self._cpp_server_url = f"http://127.0.0.1:{self._cpp_server_port}"
        self._output_dir = os.path.join(
            llamacpp_root, f"tools/omni/output_{self._cpp_server_port}"
        )
        self._temp_dir = tempfile.mkdtemp(prefix="judge_omni_")
        self._http: Optional[httpx.Client] = None
        self._server = CppServerProcess(
            llamacpp_root=llamacpp_root,
            model_path=os.path.join(model_dir, llm_model),
            port=self._cpp_server_port,
            gpu_id=gpu_id,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            output_dir=self._output_dir,
            log_path=log_path,
        )

        self._duplex_chunk_counter: int = 0
        self._sent_wav_files: Set[str] = set()
        self._duplex_length_penalty: float = 1.1
        self._last_duplex_mode: Optional[bool] = None
        self._last_media_type: int = 2
        self._last_lang: str = "zh"
        self._last_break_time: float = 0.0
        self._round_number: int = 0

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def cpp_port(self) -> int:
        return self._cpp_server_port

    @property
    def base_url(self) -> str:
        return self._cpp_server_url

    @property
    def sent_wav_files(self) -> Set[str]:
        return self._sent_wav_files


    def start(self, duplex_mode: bool = True) -> None:
        # trust_env=False：避免系统代理拦截 localhost
        self._http = httpx.Client(
            timeout=httpx.Timeout(600.0, connect=30.0),
            trust_env=False,
        )
        self._server.start()
        self._call_omni_init(media_type=2, duplex_mode=duplex_mode)
        self._last_duplex_mode = duplex_mode
        self._last_media_type = 2

    def stop(self) -> None:
        """停 server + 关 http + 清 temp。"""
        try:
            self.duplex_stop()
        except Exception:
            pass
        self._server.stop()
        if self._http is not None:
            self._http.close()
            self._http = None
        if os.path.exists(self._temp_dir):
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def full_reinit(self) -> None:
        self._round_number = 0
        self._last_duplex_mode = None
        self._last_media_type = 2
        t_break = self._last_break_time
        if t_break > 0.0:
            flag_paths = [
                os.path.join(self._output_dir, "generation_done.flag"),
                os.path.join(self._output_dir, "tts_wav", "generation_done.flag"),
            ]

            def _flag_exists() -> bool:
                latest_round = wav_poller.find_latest_round_dir(self._output_dir)
                if latest_round:
                    latest_flag = os.path.join(
                        latest_round, "tts_wav", "generation_done.flag"
                    )
                    if latest_flag not in flag_paths:
                        flag_paths.append(latest_flag)
                return any(os.path.exists(p) for p in flag_paths)

            if not _flag_exists():
                for _ in range(20):
                    time.sleep(0.5)
                    if _flag_exists():
                        break
                else:
                    logger.warning(
                        "full_reinit: T2W completion flag not seen within 10s; "
                        "checked_paths=%s",
                        flag_paths,
                    )
        try:
            logger.info("full_reinit: stopping llama-server...")
            self._server.stop()
            logger.info("full_reinit: restarting llama-server...")
            self._server.start()
            self._call_omni_init(media_type=2, duplex_mode=True)
            self._last_duplex_mode = True
            self._last_media_type = 2
            logger.info("full_reinit: omni context re-initialized successfully")
        except Exception as e:
            logger.error("full_reinit failed: %s", e, exc_info=True)
            raise

    # HTTP: omni_init / prefill / decode

    def _call_omni_init(
        self,
        media_type: int = 2,
        duplex_mode: bool = True,
        lang: Optional[str] = None,
        system_content: Any = None,
    ) -> None:
        assert self._http is not None
        tts_bin_dir = os.path.join(self.model_dir, "tts")
        os.makedirs(self._output_dir, exist_ok=True)

        req_body: Dict[str, Any] = {
            "media_type": media_type,
            "use_tts": True,
            "duplex_mode": duplex_mode,
            "model_dir": self.model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": "gpu:0",
            "output_dir": self._output_dir,
            "seed": sampler_seed(),
        }

        if self.ref_audio_path and os.path.exists(self.ref_audio_path):
            req_body["voice_audio"] = self.ref_audio_path

        effective_lang = lang or self._last_lang
        ps = prompts.build_prompts_from_content(
            system_content, duplex_mode, effective_lang
        )
        req_body["voice_clone_prompt"] = ps["voice_clone_prompt"]
        req_body["assistant_prompt"] = ps["assistant_prompt"]
        if lang:
            self._last_lang = lang

        logger.info(
            "Calling omni_init: media_type=%s duplex=%s lang=%s seed=%s",
            media_type,
            duplex_mode,
            effective_lang,
            req_body["seed"],
        )
        resp = self._http.post(
            f"{self._cpp_server_url}/v1/stream/omni_init",
            json=req_body,
            timeout=120.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"omni_init failed: {resp.text}")
        logger.info("omni_init success: %s", resp.json())

    def _apply_sampling_knobs(self, sampling: Optional[Dict[str, Any]]) -> None:
        if not sampling:
            return
        assert self._http is not None
        body: Dict[str, Any] = {}
        for key in _CPP_SAMPLING_KEYS:
            if key in sampling and sampling[key] is not None:
                body[key] = sampling[key]
        if not body:
            return
        resp = self._http.post(
            f"{self._cpp_server_url}/v1/stream/update_session_config",
            json=body,
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"update_session_config(sampling) failed: {resp.text}")
        logger.info("applied duplex sampling knobs: %s -> %s", body, resp.json())

    def _call_prefill(
        self,
        audio_path: str,
        img_path: str,
        cnt: int,
        max_slice_nums: int = -1,
        text: str = "",
    ) -> None:
        assert self._http is not None
        req_body: Dict[str, Any] = {
            "audio_path_prefix": audio_path,
            "img_path_prefix": img_path,
            "cnt": cnt,
        }
        if max_slice_nums > 0:
            req_body["max_slice_nums"] = max_slice_nums
        if text:
            req_body["text"] = text

        resp = self._http.post(
            f"{self._cpp_server_url}/v1/stream/prefill",
            json=req_body,
            timeout=30.0,
        )
        if resp.status_code != 200:
            logger.error("prefill failed (cnt=%s): %s", cnt, resp.text)
            return

    # duplex API

    def duplex_prepare(
        self,
        ref_audio_path: Optional[str] = None,
        length_penalty: float = 1.1,
        sampling: Optional[Dict[str, Any]] = None,
        media_type: int = 2,
        lang: Optional[str] = None,
    ) -> str:
        wav_poller.reset_output_dir(self._output_dir)
        self._duplex_chunk_counter = 0
        self._round_number = 0
        self._sent_wav_files = set()
        self._duplex_length_penalty = float(length_penalty)

        self._last_duplex_mode = True
        self._last_media_type = media_type
        if lang:
            self._last_lang = lang

        if sampling:
            self._apply_sampling_knobs(sampling)

        ref_for_sys = ""
        if ref_audio_path and os.path.exists(ref_audio_path):
            ref_for_sys = ref_audio_path
        elif self.ref_audio_path and os.path.exists(self.ref_audio_path):
            ref_for_sys = self.ref_audio_path
        self._call_prefill(ref_for_sys, "", 0)
        self._duplex_chunk_counter = 1
        logger.info(
            "duplex_prepare: system+ref prefill(cnt=0) done (ref=%s); user chunks start at cnt=1",
            display_path(ref_for_sys) if ref_for_sys else "(empty)",
        )

        os.makedirs(os.path.join(self._output_dir, "tts_wav"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "tts_txt"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "llm_debug"), exist_ok=True)
        return "Streaming Duplex Conversation."

    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[list] = None,
        max_slice_nums: int = 1,
    ) -> PrefillResult:
        cnt = self._duplex_chunk_counter
        self._duplex_chunk_counter += 1

        temp_audio = ""
        if audio_waveform is not None and len(audio_waveform) > 0:
            temp_audio = media.save_audio_to_temp(
                audio_waveform, self._temp_dir, f"duplex_{cnt}"
            )

        temp_image = ""
        n_vision_images = 0
        if frame_list:
            temp_image = media.save_pil_image_to_temp(
                frame_list[0], self._temp_dir, f"duplex_{cnt}"
            )
            n_vision_images = 1

        self._call_prefill(temp_audio, temp_image, cnt, max_slice_nums)

        if frame_list and len(frame_list) > 1:
            for i, frame in enumerate(frame_list[1:], 1):
                extra_img = media.save_pil_image_to_temp(
                    frame, self._temp_dir, f"duplex_{cnt}_f{i}"
                )
                self._call_prefill("", extra_img, cnt + i, max_slice_nums)
                n_vision_images += 1

        media.cleanup_temp_files(temp_audio, temp_image)
        return PrefillResult(n_vision_images=n_vision_images, cnt=cnt)

    def duplex_generate(self, force_listen: bool = False) -> DuplexGenerateResult:
        assert self._http is not None
        _ = force_listen
        t0 = time.perf_counter()

        resp = self._http.post(
            f"{self._cpp_server_url}/v1/stream/decode",
            json={
                "stream": True,
                "length_penalty": float(self._duplex_length_penalty),
            },
            timeout=600.0,
        )

        is_listen = True
        end_of_turn = False
        turn_idle = False
        texts: List[str] = []
        metrics: Dict[str, Any] = {}

        if resp.status_code == 200:
            for line in resp.text.splitlines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json_loads(data_str)
                except Exception:
                    continue

                if event.get("event") == "metrics":
                    metrics = event
                    continue

                if "is_listen" in event:
                    is_listen = event["is_listen"]
                if "end_of_turn" in event:
                    end_of_turn = event["end_of_turn"]
                if event.get("turn_idle"):
                    turn_idle = True
                if event.get("text"):
                    texts.append(event["text"])
                if event.get("content"):
                    texts.append(event["content"])

        text = "".join(texts)
        cost_all_ms = (time.perf_counter() - t0) * 1000

        def _f(key: str) -> Optional[float]:
            v = metrics.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        stage_cnt = metrics.get("cnt")
        try:
            stage_cnt_i = int(stage_cnt) if stage_cnt is not None else None
        except (TypeError, ValueError):
            stage_cnt_i = None

        return DuplexGenerateResult(
            is_listen=is_listen,
            text=text,
            end_of_turn=end_of_turn,
            turn_idle=turn_idle,
            current_time=self._duplex_chunk_counter,
            cost_all_ms=round(cost_all_ms, 1),
            cost_llm_ms=_f("cost_llm_ms"),
            cost_tts_ms=_f("cost_tts_ms"),
            cost_token2wav_ms=_f("cost_token2wav_ms"),
            vpm_ms=_f("vpm_ms"),
            apm_ms=_f("apm_ms"),
            llm_prefill_ms=_f("llm_prefill_ms"),
            stage_cnt=stage_cnt_i,
        )

    def duplex_stop(self) -> None:
        self._last_break_time = time.time()
        if self._http is None:
            return
        try:
            self._http.post(
                f"{self._cpp_server_url}/v1/stream/break",
                json={"reason": "duplex_stop"},
                timeout=10.0,
            )
        except Exception as e:
            logger.warning("duplex_stop break call failed: %s", e)

    def collect_wav_nowait(self):
        return wav_poller.collect_wav_output_nowait(
            self._output_dir, self._sent_wav_files
        )


def json_loads(s: str) -> Any:
    import json

    return json.loads(s)
