import pandas as pd
import matplotlib.pyplot as plt

exp_name = "fixed_exp_torch_vs_deepspeed_vs_eager"
path = f"experiments/{exp_name}/"
df = pd.read_csv(path+"final_output.csv")

# === Plot 1: Time per Iteration ===
df_time = df.pivot(index="seq_len", columns="compile", values="time_per_iter")
ax1 = df_time.plot(marker='o')
ax1.set_xlabel("Sequence Length")
ax1.set_ylabel("Time per Iteration (s)")
ax1.set_title("Time per Iteration vs Sequence Length")
ax1.grid(True)
ax1.set_xticks(df_time.index)  # Force x-axis ticks to be the actual data points
plt.tight_layout()
plt.savefig(path+"time_per_iter_vs_seq_len.png")

# === Plot 2: Peak Memory ===
df_memory = df.pivot(index="seq_len", columns="compile", values="peak_memory")
ax2 = df_memory.plot(marker='o')
ax2.set_xlabel("Sequence Length")
ax2.set_ylabel("Peak Memory (GB)")
ax2.set_title("Peak Memory vs Sequence Length")
ax2.grid(True)
ax2.set_xticks(df_memory.index)
plt.tight_layout()
plt.savefig(path+"peak_memory_vs_seq_len.png")
