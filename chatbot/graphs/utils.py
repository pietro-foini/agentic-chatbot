from langchain.agents.middleware import ModelRequest


def extract_model_name(request: ModelRequest, msg) -> str:
    """Extract the model name from the request or the message metadata."""

    if hasattr(request, "model"):
        model_obj = request.model
        name = getattr(model_obj, "model_name", None) or getattr(model_obj, "model", None)
        if name:
            return name

    metadata = getattr(msg, "response_metadata", {}) or {}

    return metadata.get("model_name") or metadata.get("model", "")
