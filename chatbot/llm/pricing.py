def compute_cost(in_tokens: int, out_tokens: int) -> float:
    """Compute cost of tokens"""

    total_cost = (in_tokens / 1_000_000 * 0.10) + (out_tokens / 1_000_000 * 0.40)

    return total_cost
