"""
Virtual Expert System for MoE Models.

This subpackage provides a plugin-based framework for adding virtual experts
to MoE models. Virtual experts are external tools (Python functions, APIs,
databases, etc.) that can be routed to by the MoE router.

Virtual expert base class is provided by chuk-virtual-expert package.

Structure:
    - base.py: Re-exports VirtualExpert from chuk-virtual-expert + inference types
    - registry.py: VirtualExpertRegistry for managing plugins
    - router.py: VirtualRouter that wraps MoE routers
    - wrapper.py: VirtualMoEWrapper main interface
    - plugins/: Built-in plugin implementations
        - math.py: MathExpert for arithmetic

Example - Using built-in math expert:
    >>> from chuk_lazarus.inference import VirtualMoEWrapper
    >>>
    >>> wrapper = VirtualMoEWrapper(model, tokenizer)
    >>> wrapper.calibrate()
    >>>
    >>> result = wrapper.solve("127 * 89 = ")
    >>> print(result.answer)  # "11303"

Example - Using TimeExpert from chuk-virtual-expert-time:
    >>> from chuk_virtual_expert_time import TimeExpert
    >>> from chuk_virtual_expert import adapt_expert
    >>>
    >>> expert = TimeExpert()
    >>> adapter = adapt_expert(expert)
    >>> wrapper.register_plugin(adapter)
"""

from importlib import import_module

_EXPORTS = {
    "VirtualExpert": ("._optional", "VirtualExpert"),
    "VirtualExpertPlugin": (".base", "VirtualExpertPlugin"),
    "VirtualExpertResult": (".base", "VirtualExpertResult"),
    "VirtualExpertAnalysis": (".base", "VirtualExpertAnalysis"),
    "VirtualExpertApproach": (".base", "VirtualExpertApproach"),
    "InferenceResult": (".base", "InferenceResult"),
    "VirtualExpertAction": (".cot_rewriter", "VirtualExpertAction"),
    "CoTRewriter": (".cot_rewriter", "CoTRewriter"),
    "FewShotCoTRewriter": (".cot_rewriter", "FewShotCoTRewriter"),
    "DirectCoTRewriter": (".cot_rewriter", "DirectCoTRewriter"),
    "RoutingDecision": (".base", "RoutingDecision"),
    "RoutingTrace": (".base", "RoutingTrace"),
    "VirtualExpertRegistry": (".registry", "VirtualExpertRegistry"),
    "get_default_registry": (".registry", "get_default_registry"),
    "VirtualRouter": (".router", "VirtualRouter"),
    "VirtualMoEWrapper": (".wrapper", "VirtualMoEWrapper"),
    "create_virtual_expert_wrapper": (".wrapper", "create_virtual_expert_wrapper"),
    "VirtualDenseRouter": (".dense_wrapper", "VirtualDenseRouter"),
    "VirtualDenseWrapper": (".dense_wrapper", "VirtualDenseWrapper"),
    "create_virtual_dense_wrapper": (".dense_wrapper", "create_virtual_dense_wrapper"),
    "MathExpert": (".plugins.math", "MathExpert"),
    "MathExpertPlugin": (".plugins.math", "MathExpertPlugin"),
    "SafeMathEvaluator": (".plugins.math", "SafeMathEvaluator"),
}

__all__ = [
    # Base classes (from chuk-virtual-expert)
    "VirtualExpert",
    "VirtualExpertPlugin",  # Alias for backwards compat
    "VirtualExpertResult",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    "InferenceResult",
    # CoT rewriting
    "VirtualExpertAction",
    "CoTRewriter",
    "FewShotCoTRewriter",
    "DirectCoTRewriter",
    # Routing trace (verbose output)
    "RoutingDecision",
    "RoutingTrace",
    # Registry
    "VirtualExpertRegistry",
    "get_default_registry",
    # Router (MoE)
    "VirtualRouter",
    # Wrapper (MoE)
    "VirtualMoEWrapper",
    "create_virtual_expert_wrapper",
    # Router (Dense)
    "VirtualDenseRouter",
    # Wrapper (Dense)
    "VirtualDenseWrapper",
    "create_virtual_dense_wrapper",
    # Built-in plugins
    "MathExpert",
    "MathExpertPlugin",  # Alias for backwards compat
    "SafeMathEvaluator",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
