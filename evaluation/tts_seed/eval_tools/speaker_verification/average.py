import sys

import numpy as np


infile = sys.argv[1]
outfile = sys.argv[2]

fout = open(outfile, "w")
scores = []
for line in open(infile, "r").readlines():
    item, score = line.strip().split("\t")
    scores.append(float(score))
    fout.write(line)

res = round(np.mean(np.array(scores)), 3)
res_var = round(np.var(np.array(scores)), 3)
fout.write(f"ASV: {res}\n")
fout.write(f"ASV-var: {res_var}\n")
print(f"ASV: {res}")
print(f"ASV-var: {res_var}")
fout.close()

all_outfile = infile.replace("merge.out", "eval_result.out")
with open(all_outfile, "a") as f:
    f.write(f"ASV: {res}\n")
    f.write(f"ASV-var: {res_var}\n")

print("result written to", all_outfile)
