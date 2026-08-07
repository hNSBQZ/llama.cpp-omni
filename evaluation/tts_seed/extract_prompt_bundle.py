#!/usr/bin/env python3
"""
从参考音频提取 prompt_bundle 三个文件（spk、tokens、mel），
供 C++ Token2Wav 使用。

支持两种模式：
  1. 单文件模式：--ref-audio + --output-dir
  2. 批量模式：  --batch-list (每行: wav_path \t output_dir)

用法:
    # 单文件
    python extract_prompt_bundle.py \
        --ref-audio ref.wav \
        --model-dir /path/to/onnx_models \
        --output-dir /path/to/bundle_out

    # 批量
    python extract_prompt_bundle.py \
        --batch-list batch.tsv \
        --model-dir /path/to/onnx_models
"""

import argparse
import os
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import s3tokenizer
import onnxruntime
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm


# ==================== mel 频谱计算 ====================

_mel_basis = {}
_hann_window = {}


def mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000,
                    hop_size=480, win_size=1920, fmin=0, fmax=8000):
    global _mel_basis, _hann_window
    key_mel = f"{fmax}_{y.device}"
    key_win = str(y.device)
    if key_mel not in _mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[key_mel] = torch.from_numpy(mel).float().to(y.device)
        _hann_window[key_win] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=_hann_window[key_win], center=False,
        pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis[key_mel], spec)
    return torch.log(torch.clamp(spec, min=1e-5))


# ==================== 提取单个音频 ====================

def extract_single(ref_audio: str, output_dir: str,
                   s3_tokenizer, spk_session, device: str = "cuda"):
    """提取单个参考音频的 prompt_bundle，使用已加载的模型。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. speech tokens
    audio_16k = s3tokenizer.load_audio(ref_audio, sr=16000)
    mels_s3 = s3tokenizer.log_mel_spectrogram(audio_16k)
    mels_s3, mels_s3_lens = s3tokenizer.padding([mels_s3])
    tokens, _ = s3_tokenizer.quantize(
        mels_s3.to(device), mels_s3_lens.to(device)
    )

    # 2. speaker embedding
    spk_feat = kaldi.fbank(audio_16k.unsqueeze(0), num_mel_bins=80, dither=0,
                           sample_frequency=16000)
    spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
    spk_emb = torch.tensor(
        spk_session.run(
            None,
            {spk_session.get_inputs()[0].name: spk_feat.unsqueeze(0).cpu().numpy()}
        )[0],
        device=device,
    )

    # 3. mel spectrogram (24kHz)
    audio_raw, sr = torchaudio.load(ref_audio)
    audio_raw = audio_raw.mean(dim=0, keepdim=True)
    if sr != 24000:
        audio_raw = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(audio_raw)
    prompt_mel = mel_spectrogram(audio_raw).transpose(1, 2).squeeze(0)
    prompt_mels = prompt_mel.unsqueeze(0).to(device)

    UP_RATE = 2
    prompt_mels = torch.nn.functional.pad(
        prompt_mels,
        (0, 0, 0, tokens.shape[1] * UP_RATE - prompt_mels.shape[1]),
        mode="replicate",
    )

    # 4. 截断 mel 并保存
    T = tokens.shape[1]
    PRE_LOOKAHEAD = 3
    expect_T_mel = (T - PRE_LOOKAHEAD) * UP_RATE

    spk_emb.squeeze().cpu().float().numpy().tofile(
        os.path.join(output_dir, "spk_f32.bin"))
    tokens.squeeze().cpu().int().numpy().tofile(
        os.path.join(output_dir, "prompt_tokens_i32.bin"))
    prompt_mels[:, :expect_T_mel, :].squeeze(0).cpu().float().numpy() \
        .flatten().tofile(os.path.join(output_dir, "prompt_mel_btc_f32.bin"))

    return T, expect_T_mel


def load_models(model_dir: str, device: str = "cuda"):
    """加载 S3 tokenizer 和 CAM++ 说话人识别模型，返回 (s3_tokenizer, spk_session)。"""
    s3_tok = s3tokenizer.load_model(
        os.path.join(model_dir, "speech_tokenizer_v2_25hz.onnx")
    ).to(device).eval()

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    spk_sess = onnxruntime.InferenceSession(
        os.path.join(model_dir, "campplus.onnx"),
        sess_options=option, providers=["CPUExecutionProvider"],
    )
    return s3_tok, spk_sess


def main():
    parser = argparse.ArgumentParser(description="从参考音频提取 prompt_bundle 文件")
    parser.add_argument("--ref-audio", "-r", help="单文件模式: 参考音频 WAV 路径")
    parser.add_argument("--output-dir", "-o", help="单文件模式: 输出目录")
    parser.add_argument("--batch-list", "-b",
                        help="批量模式: TSV 文件，每行 wav_path\\toutput_dir")
    parser.add_argument("--model-dir", "-m", required=True,
                        help="包含 speech_tokenizer_v2_25hz.onnx 和 campplus.onnx 的目录")
    parser.add_argument("--device", "-d", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在 spk_f32.bin 的输出目录")
    args = parser.parse_args()

    if not args.ref_audio and not args.batch_list:
        parser.error("必须指定 --ref-audio 或 --batch-list")

    print(f"Loading models from {args.model_dir} ...")
    s3_tok, spk_sess = load_models(args.model_dir, args.device)
    print("Models loaded.")

    if args.batch_list:
        with open(args.batch_list, "r") as f:
            tasks = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    print(f"WARNING: skipping malformed line: {line}")
                    continue
                tasks.append((parts[0], parts[1]))

        print(f"Batch mode: {len(tasks)} audio files to process")
        success, skip, fail = 0, 0, 0
        for wav_path, out_dir in tqdm(tasks, desc="Extracting prompt_bundles"):
            if args.skip_existing and os.path.exists(os.path.join(out_dir, "spk_f32.bin")):
                skip += 1
                continue
            try:
                T, T_mel = extract_single(wav_path, out_dir, s3_tok, spk_sess, args.device)
                success += 1
            except Exception as e:
                print(f"ERROR processing {wav_path}: {e}")
                fail += 1

        print(f"Done! success={success}, skipped={skip}, failed={fail}")

    else:
        if not args.output_dir:
            parser.error("单文件模式需要指定 --output-dir")
        T, T_mel = extract_single(args.ref_audio, args.output_dir,
                                  s3_tok, spk_sess, args.device)
        print(f"Done! T_token={T}, T_mel={T_mel}")
        print(f"Files saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
