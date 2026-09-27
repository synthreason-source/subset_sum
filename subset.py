import math
import random
import heapq
import numpy as np

def number_to_holographic_tensor(num_str, max_len):
    """
    Projects a decimal number string into a 3D holographic cardinal tensor
    across positional (X), phase (Y), and depth (Z) axes.
    """
    padded = num_str.zfill(max_len)
    tensor = np.zeros((max_len, 10, 3), dtype=float)
    
    for idx, char in enumerate(padded):
        digit = int(char)
        norm_pos = idx / max(1, max_len - 1)
        
        # X-axis: Positional magnitude
        tensor[idx, digit, 0] = 1.0 - norm_pos
        # Y-axis: Decimal phase encoding (trigonometric modulation)
        tensor[idx, digit, 1] = np.sin(2.0 * np.pi * (digit / 10.0))
        # Z-axis: Interference depth index
        tensor[idx, digit, 2] = np.cos(2.0 * np.pi * norm_pos)
        
    return tensor

def compute_holographic_resonance(partial_val, target_str, max_len, target_tensor, target_norm):
    """
    Computes structural holographic resonance between a candidate partial sum
    and the target decimal tensor field.
    """
    p_str = str(partial_val)
    p_tensor = number_to_holographic_tensor(p_str, max_len)
    p_norm = np.linalg.norm(p_tensor)
    
    if p_norm == 0 or target_norm == 0:
        return 0.0
        
    # Normalized inner product (structural similarity/interference score)
    return np.sum(target_tensor * p_tensor) / (target_norm * p_norm)

def _materialize(node):
    """Walk an immutable path linked-list back to a plain Python list."""
    out = []
    while node is not None:
        val, node = node
        out.append(val)
    out.reverse()
    return out

def holographic_equation_beam_search(nums, target, r, c=2, max_beam_width=2000):
    """
    Equation-driven beam search enhanced with holographic decimal tensor 
    projections for structural candidate ranking and path pruning.
    """
    n = len(nums)
    theoretical_cap = n ** c
    cap = min(theoretical_cap, max_beam_width)

    target_str = str(target)
    max_len = len(target_str) + 2
    target_tensor = number_to_holographic_tensor(target_str, max_len)
    target_norm = np.linalg.norm(target_tensor)

    # beam: list of (partial_sum, path_node)
    beam = [(0, None)]

    for k in range(1, n + 1):
        x = nums[k - 1]

        # Deduplicate by partial sum as we expand
        seen = {}
        for partial, node in beam:
            # Exclude: path unchanged, O(1)
            if partial not in seen:
                seen[partial] = node
            # Include: O(1) cons cell allocation
            new_partial = partial + x
            if new_partial <= target and new_partial not in seen:
                seen[new_partial] = (x, node)

        if target in seen:
            return _materialize(seen[target]), k

        # Equation-driven pruning width calculation
        log_width = n * math.log(2) + k * math.log(1 - r)
        if log_width > math.log(cap):
            width = cap
        else:
            width = max(1, min(cap, math.ceil(math.exp(log_width))))

        items = list(seen.items())
        if len(items) <= width:
            beam = items
        else:
            # Rank candidates using a hybrid metric: 
            # Numerical distance combined with holographic decimal tensor resonance
            scored_items = []
            for partial, node in items:
                dist_penalty = abs(target - partial)
                # Higher resonance reduces the sorting score (making it more favorable)
                resonance = compute_holographic_resonance(partial, target_str, max_len, target_tensor, target_norm)
                hybrid_score = dist_penalty - (resonance * 10.0)
                scored_items.append((hybrid_score, partial, node))
            
            # Pick width-best candidates via heap selection
            best_items = heapq.nsmallest(width, scored_items)
            beam = [(partial, node) for _, partial, node in best_items]

    return None, n

def measure_r(nums, sample_size=24, subset_fraction=5, seed=None):
    """Automatically estimates r for the dataset via pilot sampling."""
    rng = random.Random(seed)
    sample_size = min(sample_size, len(nums))
    sample = rng.sample(nums, sample_size)
    k = max(1, min(subset_fraction, sample_size))
    sample_target = sum(rng.sample(sample, k))

    n = sample_size
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sample[i]

    nodes = 0
    stack = [(0, 0)]
    while stack:
        i, partial = stack.pop()
        nodes += 1
        if partial == sample_target:
            break
        if i == n or partial > sample_target or partial + suffix[i] < sample_target:
            continue
        stack.append((i + 1, partial))
        stack.append((i + 1, partial + sample[i]))

    total = 2 ** n
    ratio = max(nodes / total, 1e-300)
    r = 1 - math.exp(math.log(ratio) / n)
    return min(max(r, 1e-6), 1 - 1e-6), nodes, sample_size, sample_target

def run(label, nums, target, c=2, max_beam_width=2000, seed=None):
    r, sample_nodes, sample_n, sample_target = measure_r(nums, seed=seed)
    print(f"=== {label} (n={len(nums)}, target={target}, c={c}, max_beam_width={max_beam_width}) ===")
    print(f"auto-measured r: {r:.4f} (pilot nodes: {sample_nodes})")

    result, steps_used = holographic_equation_beam_search(nums, target, r, c, max_beam_width)
    print(f"holographic equation beam search: {'found ' + str(result) if result else 'FAILED'} (ran {steps_used} steps)\n")

if __name__ == "__main__":
    random.seed(42)
    n = 50000000000
    hard_nums = [random.randint(1, 100_000) for _ in range(n)]
    hard_target = sum(random.sample(hard_nums, 150))
    
    run("Holographic Beam Search Test", hard_nums, hard_target, max_beam_width=1000, seed=42)
