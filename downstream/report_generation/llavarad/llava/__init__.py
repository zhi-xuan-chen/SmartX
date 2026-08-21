"""LLaVA-Rad package with lazy model exports."""

__all__ = ["LlavaLlamaForCausalLM"]


def __getattr__(name):
    if name == "LlavaLlamaForCausalLM":
        from .model import LlavaLlamaForCausalLM

        return LlavaLlamaForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
