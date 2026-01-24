import os
import torch

TENSOR_DIR_1 = "tensors"
TENSOR_DIR_2 = "ulyssess_tensors"
ATOL = 1e-4

# Define comparison order
ORDERED_PATTERNS = [
    # "query_0_before.pt", "query_1_before.pt",
    # "key_0_before.pt", "key_1_before.pt",
    # "value_0_before.pt", "value_1_before.pt",
    "query_0_after.pt", "query_1_after.pt",
    "key_0_after.pt", "key_1_after.pt",
    "value_0_after.pt", "value_1_after.pt",
    "out_0_before.pt", "out_1_before.pt",
    "out_0_after.pt", "out_1_after.pt",
]

def compare_tensor_values_only(file_name, atol=1e-2):
    path1 = os.path.join(TENSOR_DIR_1, file_name)
    path2 = os.path.join(TENSOR_DIR_2, file_name)
    
    t1 = torch.load(path1)
    t2 = torch.load(path2)

    assert t1.shape == t2.shape, f"[SHAPE MISMATCH] {file_name}: {t1.shape} vs {t2.shape}"

    flat1 = t1.flatten().sort()[0]
    flat2 = t2.flatten().sort()[0]

    if not torch.allclose(flat1, flat2, atol=atol):
        print(f"[SORTED VALUE MISMATCH] {file_name}")
        diffs = torch.abs(flat1 - flat2)
        max_diff = diffs.max().item()
        max_idx = diffs.argmax().item()
        print(f"  -> Max absolute difference: {max_diff:.6f} at index {max_idx}")
        print(f"  -> Values: tensor1 = {flat1[max_idx].item():.6f}, tensor2 = {flat2[max_idx].item():.6f}")

        # Optional: print top 5 largest mismatches
        # topk_vals, topk_indices = torch.topk(diffs, k=5)
        # for i in range(5):
        #     idx = topk_indices[i].item()
        #     print(f"     [{i}] Δ = {topk_vals[i].item():.6f} | t1 = {flat1[idx].item():.6f}, t2 = {flat2[idx].item():.6f}")
    else:
        print(f"[VALUES MATCH (SORTED)] {file_name}")

def compare_tensors(file_name):
    path1 = os.path.join(TENSOR_DIR_1, file_name)
    path2 = os.path.join(TENSOR_DIR_2, file_name)
    
    t1 = torch.load(path1)
    t2 = torch.load(path2)
    
    assert t1.stride() == t2.stride(), f"[STRIDE MISMATCH] {file_name}: {t1.stride()} vs {t2.stride()}"
    assert t1.shape == t2.shape, f"[SHAPE MISMATCH] {file_name}: {t1.shape} vs {t2.shape}"
    
    if not torch.allclose(t1, t2, atol=ATOL):
        print(f"[VALUE MISMATCH] {file_name}")
    else:
        print(f"[OK] {file_name}")

def main():
    print("Comparing tensors between 'tensors/' and 'ulysses_tensors/'...\n")
    for fname in ORDERED_PATTERNS:
        compare_tensors(fname)
        
if __name__ == "__main__":
    main()


    
    