def compute_cost(model_name: str, in_tokens: int, out_tokens: int) -> float:
    """Compute cost of tokens"""

    if model_name == "gemini-2.5-flash-lite":
        total_cost = (in_tokens / 1_000_000 * 0.10) + (out_tokens / 1_000_000 * 0.40)
    elif model_name == "gemini-3.8-flash":
        total_cost = (in_tokens / 1_000_000 * 0.75) + (out_tokens / 1_000_000 * 3.75)
    elif model_name == "gemini-3.5-flash-lite":
        total_cost = (in_tokens / 1_000_000 * 0.30) + (out_tokens / 1_000_000 * 2.50)
    else:
        total_cost = 0

    return total_cost
