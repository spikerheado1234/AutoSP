import os, sys, json
import itertools
import numpy as np
from tqdm import tqdm

os.system("rm slurm_jobs/*")

with open("sample.slurm", "r") as f:
    original = f.read()

job_list = []
base_cmd = "bash run_ulysses.sh"
sequences = [40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496]
compile_args = ["deepcompile", "compile", "eager"]
exp_name = "fixed_exp_torch_vs_deepspeed_vs_eager"

for compile_arg in compile_args:
    for seq_len in sequences:
        if compile_arg == "eager" and seq_len >= 81920:
            continue
        cmd = base_cmd + " " + str(seq_len // 2) + " " + compile_arg + " " + exp_name
        job_list.append(cmd)

for idx, job in enumerate(tqdm(job_list, desc="Creating SLURM jobs")):
    job_content = original + job
    with open(f"slurm_jobs/job_{idx}.slurm", "w") as f:
        f.write(job_content)