import matplotlib.pyplot as plt
import pandas as pd

partition = 0
sequence = 512

# Load CSV files
compile_df = pd.read_csv(f"logs/loss_compile_{sequence}_{partition}.csv")
eager_df = pd.read_csv(f"logs/loss_eager_{sequence}_{partition}.csv")

compile_df = compile_df[(compile_df["step"] > 50) & (compile_df["step"] < 100)]
eager_df = eager_df[(eager_df["step"] > 50) & (eager_df["step"] < 100)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(compile_df["step"], compile_df["loss"], label="torch.compile", linewidth=2)
plt.plot(eager_df["step"], eager_df["loss"], label="eager", linewidth=2, linestyle="--")

# Labels and title
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to file
plt.savefig(f"loss_comparison_{partition}.png")
print(f"Saved plot to loss_comparison_{partition}.png")
