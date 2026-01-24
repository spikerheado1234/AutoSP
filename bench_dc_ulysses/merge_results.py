import os
import pandas as pd

def merge_csv_files(input_folder="experiments/fixed_exp_torch_vs_deepspeed_vs_eager/"):
    output_filename = os.path.join(input_folder, f"/u/ndani/Autocompilation/bench_dc_ulysses/{input_folder}final_output.csv")
    
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    
    if not csv_files:
        print(f"No files found")
        return
    
    df_list = [pd.read_csv(os.path.join(input_folder, file)) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    merged_df = merged_df.sort_values(by=['batch', 'seq_len', 'compile'])
    
    merged_df.to_csv(output_filename, index=False)
    print(f"Merged {len(csv_files)} files into {output_filename}")

if __name__ == "__main__":
    merge_csv_files()