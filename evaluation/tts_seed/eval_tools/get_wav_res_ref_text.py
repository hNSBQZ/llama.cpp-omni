import os
import sys

from tqdm import tqdm


metalst = sys.argv[1]
wav_dir = sys.argv[2]
wav_res_ref_text = sys.argv[3]

with open(metalst, "r") as f:
    lines = f.readlines()

with open(wav_res_ref_text, "w") as f_w:
    for line in tqdm(lines):
        parts = line.strip().split("|")
        if len(parts) == 5:
            utt, prompt_text, prompt_wav, infer_text, infer_wav = parts
        elif len(parts) == 4:
            utt, prompt_text, prompt_wav, infer_text = parts
        elif len(parts) == 2:
            utt, infer_text = parts
            prompt_wav = None
        elif len(parts) == 3:
            utt, infer_text, prompt_wav = parts
            if utt.endswith(".wav"):
                utt = utt[:-4]
        else:
            continue

        if not os.path.exists(os.path.join(wav_dir, utt + ".wav")):
            continue

        if prompt_wav is not None and not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        if len(parts) == 2:
            out_line = "|".join([os.path.join(wav_dir, utt + ".wav"), infer_text])
        else:
            out_line = "|".join([os.path.join(wav_dir, utt + ".wav"), prompt_wav, infer_text])
        f_w.write(out_line + "\n")
