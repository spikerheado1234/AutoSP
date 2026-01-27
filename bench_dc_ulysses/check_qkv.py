import os
import torch

TENSOR_DIR = "debug_tensors"
ATOL = 1e-4
RTOL = 1e-5

# Mapping: autosp tensor name -> ground truth tensor name
# All tensors are in [B, N, S, H] format
INPUT_COMPARISONS = [
    # (autosp_file, ground_truth_file, description)
    ("q_input_rank0.pt", "gt_q_input_rank0.pt", "Q input rank0"),
    ("q_input_rank1.pt", "gt_q_input_rank1.pt", "Q input rank1"),
    ("k_input_rank0.pt", "gt_k_input_rank0.pt", "K input rank0"),
    ("k_input_rank1.pt", "gt_k_input_rank1.pt", "K input rank1"),
    ("v_input_rank0.pt", "gt_v_input_rank0.pt", "V input rank0"),
    ("v_input_rank1.pt", "gt_v_input_rank1.pt", "V input rank1"),
]

OUTPUT_COMPARISONS = [
    # After all-to-all (before SDPA)
    ("q_output_rank0.pt", "gt_q_after_a2a_rank0.pt", "Q after A2A rank0"),
    ("q_output_rank1.pt", "gt_q_after_a2a_rank1.pt", "Q after A2A rank1"),
    ("k_output_rank0.pt", "gt_k_after_a2a_rank0.pt", "K after A2A rank0"),
    ("k_output_rank1.pt", "gt_k_after_a2a_rank1.pt", "K after A2A rank1"),
    ("v_output_rank0.pt", "gt_v_after_a2a_rank0.pt", "V after A2A rank0"),
    ("v_output_rank1.pt", "gt_v_after_a2a_rank1.pt", "V after A2A rank1"),
    # Final output (after output all-to-all)
    ("o_output_rank0.pt", "gt_o_output_rank0.pt", "O final output rank0"),
    ("o_output_rank1.pt", "gt_o_output_rank1.pt", "O final output rank1"),
]


def compare_tensors(autosp_file, gt_file, description):
    path1 = os.path.join(TENSOR_DIR, autosp_file)
    path2 = os.path.join(TENSOR_DIR, gt_file)
    
    # Check files exist
    if not os.path.exists(path1):
        print(f"[MISSING] {description}: {autosp_file} not found")
        return False
    if not os.path.exists(path2):
        print(f"[MISSING] {description}: {gt_file} not found")
        return False
    
    t1 = torch.load(path1, weights_only=True)
    t2 = torch.load(path2, weights_only=True)
    breakpoint()
    # Check dimensions
    if t1.shape != t2.shape:
        print(f"[SHAPE MISMATCH] {description}")
        print(f"  autosp:  {autosp_file} shape = {t1.shape}")
        print(f"  gt:      {gt_file} shape = {t2.shape}")
        return False
    
    # Check values
    if not torch.allclose(t1, t2, atol=ATOL, rtol=RTOL):
        diffs = torch.abs(t1 - t2)
        max_diff = diffs.max().item()
        mean_diff = diffs.mean().item()
        max_idx = diffs.argmax().item()
        
        # Get multi-dim index
        multi_idx = []
        idx = max_idx
        for dim in reversed(t1.shape):
            multi_idx.insert(0, idx % dim)
            idx //= dim
        
        print(f"[VALUE MISMATCH] {description}")
        print(f"  shape: {t1.shape}")
        print(f"  max diff: {max_diff:.6e} at {tuple(multi_idx)}")
        print(f"  mean diff: {mean_diff:.6e}")
        print(f"  autosp val: {t1.flatten()[max_idx].item():.6f}")
        print(f"  gt val:     {t2.flatten()[max_idx].item():.6f}")
        return False
    
    print(f"[OK] {description} - shape {t1.shape}")
    return True


def main():
    print(f"Comparing tensors in '{TENSOR_DIR}/'")
    print(f"Tolerance: atol={ATOL}, rtol={RTOL}")
    print("=" * 60)
    
    # Check inputs first
    print("\n--- INPUT COMPARISONS (before all-to-all) ---")
    input_ok = True
    for autosp_file, gt_file, desc in INPUT_COMPARISONS:
        if not compare_tensors(autosp_file, gt_file, desc):
            input_ok = False
    
    # Check outputs
    print("\n--- OUTPUT COMPARISONS (after all-to-all) ---")
    output_ok = True
    for autosp_file, gt_file, desc in OUTPUT_COMPARISONS:
        if not compare_tensors(autosp_file, gt_file, desc):
            output_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if input_ok and output_ok:
        print("SUCCESS: All tensors match!")
    else:
        if not input_ok:
            print("FAILED: Input tensors have mismatches")
        if not output_ok:
            print("FAILED: Output tensors have mismatches")


if __name__ == "__main__":
    main()