import math
import random
import heapq
import numpy as np
import struct
import os
import mmap

def number_to_holographic_tensor(num_str, max_len):
    """Projects a decimal number string into a 3D holographic cardinal tensor."""
    padded = num_str.zfill(max_len)
    tensor = np.zeros((max_len, 10, 3), dtype=float)
    
    for idx, char in enumerate(padded):
        digit = int(char)
        norm_pos = idx / max(1, max_len - 1)
        
        tensor[idx, digit, 0] = 1.0 - norm_pos
        tensor[idx, digit, 1] = np.sin(2.0 * np.pi * (digit / 10.0))
        tensor[idx, digit, 2] = np.cos(2.0 * np.pi * norm_pos)
        
    return tensor

def compute_holographic_resonance(partial_val, target_str, max_len, target_tensor, target_norm):
    """Computes structural holographic resonance against target tensor field."""
    p_str = str(partial_val)
    p_tensor = number_to_holographic_tensor(p_str, max_len)
    p_norm = np.linalg.norm(p_tensor)
    
    if p_norm == 0 or target_norm == 0:
        return 0.0
        
    return np.sum(target_tensor * p_tensor) / (target_norm * p_norm)

def _materialize(node):
    """Walk an immutable path linked-list back to a plain Python list."""
    out = []
    while node is not None:
        val, node = node
        out.append(val)
    out.reverse()
    return out

def generate_disk_dataset(filepath, n_elements, batch_size=500_000, seed=42):
    """
    Dataset Generator: Streams numbers sequentially to a compact binary file 
    on the hard disk to avoid memory bloat.
    """
    rng = random.Random(seed)
    print(f"Generating binary dataset of {n_elements:,} elements to '{filepath}'...")
    
    # Each 32-bit unsigned integer takes 4 bytes on disk
    with open(filepath, "wb") as f:
        written = 0
        while written < n_elements:
            current_batch = min(batch_size, n_elements - written)
            batch = [rng.randint(1, 100_000_000) for _ in range(current_batch)]
            # Pack as unsigned 32-bit integers ('I')
            f.write(struct.pack(f"<{current_batch}I", *batch))
            written += current_batch
            
    print(f"Dataset successfully written ({os.path.getsize(filepath) / (1024*1024):.2f} MB).\n")

def holographic_equation_beam_search_disk(filepath, n, target, r, c=2, max_beam_width=2000):
    """
    Streams numbers directly from disk using memory-mapping (mmap), 
    ensuring O(1) RAM consumption during the search execution.
    """
    theoretical_cap = n ** c
    cap = min(theoretical_cap, max_beam_width)

    target_str = str(target)
    max_len = len(target_str) + 2
    target_tensor = number_to_holographic_tensor(target_str, max_len)
    target_norm = np.linalg.norm(target_tensor)

    beam = [(0, None)]

    # Open file and map into memory for instantaneous disk streaming
    with open(filepath, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for k in range(1, n + 1):
                # Each integer is 4 bytes; calculate byte offset
                offset = (k - 1) * 4
                x = struct.unpack("<I", mm[offset:offset+4])[0]

                seen = {}
                for partial, node in beam:
                    if partial not in seen:
                        seen[partial] = node
                    new_partial = partial + x
                    if new_partial <= target and new_partial not in seen:
                        seen[new_partial] = (x, node)

                if target in seen:
                    return _materialize(seen[target]), k

                log_width = n * math.log(2) + k * math.log(1 - r)
                if log_width > math.log(cap):
                    width = cap
                else:
                    width = max(1, min(cap, math.ceil(math.exp(log_width))))

                items = list(seen.items())
                if len(items) <= width:
                    beam = items
                else:
                    scored_items = []
                    for partial, node in items:
                        dist_penalty = abs(target - partial)
                        resonance = compute_holographic_resonance(partial, target_str, max_len, target_tensor, target_norm)
                        hybrid_score = dist_penalty - (resonance * 10.0)
                        scored_items.append((hybrid_score, partial, node))
                    
                    best_items = heapq.nsmallest(width, scored_items)
                    beam = [(partial, node) for _, partial, node in best_items]

    return None, n

if __name__ == "__main__":
    db_path = "subset_numbers.bin"
    n_scale = 1_000_000_000  # Scale up freely (e.g., 50_000_000) based on your available disk space
    sample_target = 150_000_001
    auto_r = 0.35

    # Step 1: Generate dataset to disk
    if input("Generate dataset? (y): ") == "y":
        generate_disk_dataset(db_path, n_scale, seed=42)

    # Step 2: Run beam search via direct disk streaming
    print(f"Running disk-streamed beam search across {n_scale:,} records...")
    result, steps_used = holographic_equation_beam_search_disk(
        filepath=db_path,
        n=n_scale,
        target=sample_target,
        r=auto_r,
        max_beam_width=1000
    )

    if result:
        print(f"\nResult: Found subset path in {steps_used} steps!")
        print(f"Subset Elements: {result}")
        print(f"Total Sum: {sum(result)} (Target: {sample_target})")
    else:
        print(f"\nResult: FAILED (Ran {steps_used} steps)")
