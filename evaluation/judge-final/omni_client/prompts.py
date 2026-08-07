"""Duplex system prompt 模板。"""

from __future__ import annotations

from typing import Any, Dict, List

_SYSTEM_PROMPTS: Dict[tuple, Dict[str, str]] = {
    (True, "zh"): {
        "voice_clone_prompt": (
            "<|im_start|>system\nStreaming Duplex Conversation! "
            "You are a helpful assistant.\n<|audio_start|>"
        ),
        "assistant_prompt": "<|audio_end|><|im_end|>\n",
    },
    (True, "en"): {
        "voice_clone_prompt": (
            "<|im_start|>system\nStreaming Duplex Conversation! "
            "You are a helpful assistant.\n<|audio_start|>"
        ),
        "assistant_prompt": "<|audio_end|><|im_end|>\n",
    },
    (False, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>",
        "assistant_prompt": (
            "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。"
            "请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
            "<|im_end|>\n<|im_start|>user\n"
        ),
    },
    (False, "en"): {
        "voice_clone_prompt": "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>",
        "assistant_prompt": (
            "<|audio_end|>Please assist users while maintaining this voice style. "
            "Please answer the user's questions seriously and in a high quality. "
            "Please chat with the user in a highly human-like and oral style. "
            "You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
            "<|im_end|>\n<|im_start|>user\n"
        ),
    },
}


def get_system_prompts(duplex: bool, lang: str = "zh") -> Dict[str, str]:
    return _SYSTEM_PROMPTS.get((duplex, lang), _SYSTEM_PROMPTS[(duplex, "zh")])


def build_prompts_from_content(
    system_content: Any,
    duplex: bool,
    lang: str = "zh",
) -> Dict[str, str]:
    if isinstance(system_content, str):
        system_content = (
            [{"type": "text", "text": system_content}] if system_content.strip() else []
        )

    if not system_content or not isinstance(system_content, list):
        return get_system_prompts(duplex, lang)

    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    before_parts: List[str] = []
    after_parts: List[str] = []
    seen_audio = False
    for item in system_content:
        t = _get(item, "type")
        t_str = getattr(t, "value", t)
        if t_str == "audio":
            seen_audio = True
        elif t_str == "text":
            text = (_get(item, "text") or "").strip()
            if not text:
                continue
            (after_parts if seen_audio else before_parts).append(text)

    before = "\n".join(before_parts).strip()
    after = "\n".join(after_parts).strip()

    if not before and not after:
        return get_system_prompts(duplex, lang)

    voice_clone_prompt = f"<|im_start|>system\n{before}\n<|audio_start|>"
    if duplex:
        assistant_prompt = (
            f"<|audio_end|>{after}<|im_end|>\n" if after else "<|audio_end|><|im_end|>\n"
        )
    else:
        tail = (
            f"{after}<|im_end|>\n<|im_start|>user\n"
            if after
            else "<|im_end|>\n<|im_start|>user\n"
        )
        assistant_prompt = f"<|audio_end|>{tail}"

    return {
        "voice_clone_prompt": voice_clone_prompt,
        "assistant_prompt": assistant_prompt,
    }
