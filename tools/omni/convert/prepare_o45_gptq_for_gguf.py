#!/usr/bin/env python3
"""Prepare a MiniCPM-o 4.5 GPTQ HF checkout for GGUF conversion.

The omni conversion scripts in this directory expect component-specific model
folders. A GPTQ checkout is sharded and keeps the LLM in qweight/qzeros/scales
form, so this script only splits and renames tensors. It intentionally keeps
the LLM GPTQ tensors packed; use a convert_hf_to_gguf.py with GPTQ support for
the final LLM GGUF conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Callable


TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def copy_tokenizer_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        in_path = src / name
        if in_path.exists():
            shutil.copy2(in_path, dst / name)

    tok_cfg_path = dst / "tokenizer_config.json"
    if tok_cfg_path.exists():
        tok_cfg = load_json(tok_cfg_path)
        tok_cfg.pop("auto_map", None)
        tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        save_json(tok_cfg_path, tok_cfg)


def sharded_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing sharded index: {index_path}")
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid weight_map in {index_path}")
    return weight_map


def shard_names(weight_map: dict[str, str]) -> list[str]:
    return sorted(set(weight_map.values()))


def collect_tensors(
    model_dir: Path,
    weight_map: dict[str, str],
    keep: Callable[[str], bool],
    rename: Callable[[str], str],
    *,
    as_float: bool = False,
) -> dict[str, "object"]:
    from safetensors import safe_open

    tensors = {}
    for shard in shard_names(weight_map):
        shard_keys = [key for key, part in weight_map.items() if part == shard and keep(key)]
        if not shard_keys:
            continue
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            for key in shard_keys:
                tensor = f.get_tensor(key)
                if as_float:
                    tensor = tensor.float()
                tensors[rename(key)] = tensor
    return tensors


def prepare_configs(model_dir: Path, work_dir: Path, root_config: dict) -> None:
    llm_dir = work_dir / "llm"
    tts_dir = work_dir / "tts"
    vpm_dir = work_dir / "vpm"
    apm_dir = work_dir / "apm"

    llm_config = dict(root_config)
    llm_config["architectures"] = ["Qwen3ForCausalLM"]
    llm_config["model_type"] = "qwen3"
    llm_config["auto_map"] = {}
    for key in ("audio_config", "tts_config", "vision_config", "slice_config"):
        llm_config.pop(key, None)
    save_json(llm_dir / "config.json", llm_config)

    tts_config = dict(root_config["tts_config"])
    tts_config["auto_map"] = {}
    tts_config["model_type"] = "llama"
    tts_config["architectures"] = ["LlamaForCausalLM"]
    tts_config.setdefault(
        "vocab_size",
        int(tts_config.get("num_audio_tokens", 6562)) + int(tts_config.get("num_text_tokens", 152064)),
    )
    tts_config.setdefault("rms_norm_eps", 1e-6)
    tts_config.setdefault("rope_theta", 10000.0)
    tts_config.setdefault("tie_word_embeddings", False)
    save_json(tts_dir / "config.json", tts_config)

    for component_dir in (vpm_dir, apm_dir):
        component_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_dir / "config.json", component_dir / "config.json")

    copy_tokenizer_files(model_dir, llm_dir)
    copy_tokenizer_files(model_dir, tts_dir)


def prepare_llm(model_dir: Path, work_dir: Path, weight_map: dict[str, str]) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    llm_dir = work_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)

    out_weight_map: dict[str, str] = {}
    total_size = 0
    part_idx = 0
    parts = shard_names(weight_map)
    for shard in parts:
        tensors = {}
        shard_keys = [key for key, part in weight_map.items() if part == shard and key.startswith("llm.")]
        if not shard_keys:
            continue

        part_idx += 1
        out_name = f"model-{part_idx:05d}-of-{len(parts):05d}.safetensors"
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            for key in shard_keys:
                new_key = key.removeprefix("llm.")
                tensor = f.get_tensor(key)
                tensors[new_key] = tensor
                out_weight_map[new_key] = out_name
                total_size += tensor.numel() * tensor.element_size()

        save_file(tensors, llm_dir / out_name, metadata={"format": "pt"})

    save_json(
        llm_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": out_weight_map},
    )
    print(f"prepared LLM shards: {llm_dir}")


def prepare_tts(model_dir: Path, work_dir: Path, weight_map: dict[str, str]) -> None:
    from safetensors.torch import save_file

    tts_dir = work_dir / "tts"
    tensors = collect_tensors(
        model_dir,
        weight_map,
        lambda key: key.startswith("tts."),
        lambda key: key.removeprefix("tts."),
    )
    save_file(tensors, tts_dir / "model.safetensors", metadata={"format": "pt"})
    print(f"prepared TTS model: {tts_dir}")


def prepare_vpm(model_dir: Path, work_dir: Path, weight_map: dict[str, str]) -> None:
    import torch

    vpm_dir = work_dir / "vpm"
    projector = collect_tensors(
        model_dir,
        weight_map,
        lambda key: key.startswith("resampler"),
        lambda key: key,
        as_float=True,
    )
    clip = collect_tensors(
        model_dir,
        weight_map,
        lambda key: key.startswith("vpm."),
        lambda key: key.removeprefix("vpm."),
        as_float=True,
    )

    torch.save(projector, vpm_dir / "minicpmv.projector")
    torch.save(clip, vpm_dir / "minicpmv.clip")
    if (model_dir / "added_tokens.json").exists():
        (vpm_dir / "added_tokens.json").write_text("{}\n", encoding="utf-8")
    print(f"prepared VPM tensors: {vpm_dir}")


def prepare_apm(model_dir: Path, work_dir: Path, weight_map: dict[str, str]) -> None:
    import torch

    apm_dir = work_dir / "apm"
    whisper = collect_tensors(
        model_dir,
        weight_map,
        lambda key: key.startswith("apm.") or key.startswith("audio_projection_layer"),
        lambda key: key,
        as_float=True,
    )
    torch.save(whisper, apm_dir / "minicpmo.whisper")
    print(f"prepared APM tensors: {apm_dir}")


def prepare_projector_src(model_dir: Path, work_dir: Path, weight_map: dict[str, str]) -> None:
    from safetensors.torch import save_file

    projector_dir = work_dir / "projector_src"
    projector_dir.mkdir(parents=True, exist_ok=True)
    tensors = collect_tensors(
        model_dir,
        weight_map,
        lambda key: key.startswith("tts.projector_semantic."),
        lambda key: key,
    )
    save_file(tensors, projector_dir / "model.safetensors", metadata={"format": "pt"})
    print(f"prepared projector source: {projector_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="MiniCPM-o 4.5 GPTQ HF model directory")
    parser.add_argument("--work-dir", required=True, type=Path, help="intermediate component directory")
    args = parser.parse_args()

    model_dir = args.model.resolve()
    work_dir = args.work_dir.resolve()
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"missing config.json in {model_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)
    root_config = load_json(model_dir / "config.json")
    weight_map = sharded_weight_map(model_dir)

    prepare_configs(model_dir, work_dir, root_config)
    prepare_llm(model_dir, work_dir, weight_map)
    prepare_tts(model_dir, work_dir, weight_map)
    prepare_vpm(model_dir, work_dir, weight_map)
    prepare_apm(model_dir, work_dir, weight_map)
    prepare_projector_src(model_dir, work_dir, weight_map)

    print("done")


if __name__ == "__main__":
    main()
