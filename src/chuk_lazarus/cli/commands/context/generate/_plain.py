"""Plain generation — no checkpoint library, just model + prompt."""

from __future__ import annotations

import sys
from argparse import Namespace

from .._types import GenerateConfig, GenerateResult


def _plain_generate(
    config: GenerateConfig,
    args: Namespace,
    *,
    backend_name: str | None = None,
    device: str | None = None,
) -> GenerateResult:
    """Generate text from a model without any checkpoint context."""
    from .....inference import UnifiedPipeline, UnifiedPipelineConfig
    from .....inference.backends import LazarusBackend

    prompt_text = config.prompt_text
    if not prompt_text:
        print("Error: no prompt specified. Use --prompt or --prompt-file.", file=sys.stderr)
        sys.exit(1)

    # Load model
    print(f"Loading model: {config.model}", file=sys.stderr)
    pipeline_kwargs = {"verbose": False}
    pipeline_config_kwargs: dict[str, object] = {}
    if backend_name is not None:
        pipeline_config_kwargs["backend_name"] = backend_name
        pipeline_config_kwargs["backend"] = (
            LazarusBackend.CUDA if backend_name == "torch" else LazarusBackend.MLX
        )
    if device is not None:
        pipeline_config_kwargs["device"] = device
    if pipeline_config_kwargs:
        pipeline_kwargs["pipeline_config"] = UnifiedPipelineConfig(**pipeline_config_kwargs)

    pipeline = UnifiedPipeline.from_pretrained(config.model, **pipeline_kwargs)

    no_chat = getattr(args, "no_chat_template", False)
    system_prompt = getattr(args, "system_prompt", None)

    # Generate via chat or raw depending on flags
    if no_chat:
        result = pipeline.generate(
            prompt_text,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    else:
        result = pipeline.chat(
            prompt_text,
            system_message=system_prompt,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    return GenerateResult(
        response=result.text,
        tokens_generated=result.stats.output_tokens,
        context_tokens=result.stats.input_tokens,
    )
