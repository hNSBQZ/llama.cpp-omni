import sys

import numpy as np


infile = sys.argv[1]
outfile = sys.argv[2]

fout = open(outfile, "w")
fout.write("utt\twav_res\tres_wer\ttext_ref\ttext_res\tres_wer_ins\tres_wer_del\tres_wer_sub\n")
wers = []
wers_below50 = []
inses = []
deles = []
subses = []
err_num = []
total_length = []

for line in open(infile, "r").readlines():
    wav_path, wer, text_ref, text_res, inse, dele, subs = line.strip().split("\t")
    err_num.append(float(wer) * len(text_ref.split()))
    if float(wer) <= 0.5:
        wers_below50.append(float(wer))

    wers.append(float(wer))
    inses.append(float(inse))
    deles.append(float(dele))
    subses.append(float(subs))
    total_length.append(len(text_ref.split()))
    fout.write(line)

total_length = sum(total_length)
err_num = sum(err_num)
wer = round(np.mean(wers) * 100, 3)
wer_normalized = round(err_num / total_length * 100, 3)
wer_below50 = round(np.mean(wers_below50) * 100, 3)

fout.write(f"WER: {wer}%\n")
fout.write(
    f"WER_BELOW50: {wer_below50}% ,WERS_NORM_BELOW50: {len(wers_below50)}, "
    f"WER_BELOW50_RATIO: {len(wers_below50) / len(wers)}%\n"
)
fout.write(f"WER_NORMALIZED: {wer_normalized}%\n")
fout.close()

print(f"WER: {wer}%\n")
print(f"WER_NORMALIZED: {wer_normalized}%\n")
print(
    f"WER_BELOW50: {wer_below50}% ,WERS_NORM_BELOW50: {len(wers_below50)}, "
    f"WER_BELOW50_RATIO: {len(wers_below50) / len(wers)}%\n"
)

all_outfile = outfile.replace("wav_res_ref_text.wer", "eval_result.out")
with open(all_outfile, "a") as f:
    f.write(f"WER: {wer}%\n")
    f.write(f"WER_NORMALIZED: {wer_normalized}%\n")
    f.write(
        f"WER_BELOW50: {wer_below50}% ,WERS_NORM_BELOW50: {len(wers_below50)}, "
        f"WER_BELOW50_RATIO: {len(wers_below50) / len(wers)}%\n"
    )
