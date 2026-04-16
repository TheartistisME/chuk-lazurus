"""Probing analysis command handlers for introspection CLI.

This module provides thin CLI wrappers for probing analysis commands.
All business logic is delegated to the framework layer (introspection module).

IMPORTANT: CLI commands should NOT contain hardcoded sample data.
Use --calibration-file or framework-level dataset loaders instead.
"""

from __future__ import annotations

import logging
from argparse import Namespace

from .._constants import AnalysisDefaults, Delimiters, LayerDepthRatio, ProbeDefaults
from ._utils import (
    extract_arg,
    get_layer_depth_ratio,
    load_json_file,
    parse_layers,
    parse_prompts,
)

logger = logging.getLogger(__name__)


def _parse_prompt_list_preserving_trailing_space(prompts_arg: str) -> list[str]:
    """Parse prompt lists without trimming meaningful trailing spaces."""
    if prompts_arg.startswith("@"):
        with open(prompts_arg[1:]) as f:
            return [line.rstrip("\n") for line in f if line.rstrip("\n")]
    return [prompt for prompt in prompts_arg.split(Delimiters.PROMPT_SEPARATOR) if prompt]


async def introspect_metacognitive(args: Namespace) -> None:
    """Detect metacognitive strategy switch at a specific layer.

    This is a thin wrapper that:
    1. Calls MetacognitiveService.analyze() which handles all logic
    2. Formats and prints results

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.probing import MetacognitiveConfig, MetacognitiveService

    # Parse prompts from CLI
    prompts_arg = extract_arg(args, "prompts") or extract_arg(args, "problems")
    prompts = parse_prompts(prompts_arg)
    decision_layer = extract_arg(args, "decision_layer")

    config = MetacognitiveConfig(
        model=args.model,
        prompts=prompts,
        decision_layer=decision_layer,
        layer_depth_ratio=get_layer_depth_ratio(decision_layer, LayerDepthRatio.LATE),
        top_k=extract_arg(args, "top_k", AnalysisDefaults.TOP_K_LAYER),
        use_raw=extract_arg(args, "raw", False),
    )

    # Run analysis - all logic is in the service
    result = await MetacognitiveService.analyze(config)

    # Print formatted result
    print(result.to_display())


async def introspect_uncertainty(args: Namespace) -> None:
    """Analyze model's uncertainty and calibration.

    IMPORTANT: Calibration prompts should come from framework datasets
    or user-provided files, not hardcoded in the CLI.

    Args:
        args: Parsed command-line arguments
    """
    from ....datasets import load_calibration_prompts
    from ....introspection.probing import UncertaintyConfig, UncertaintyService

    # Load calibration prompts from file or use framework defaults
    calibration_file = extract_arg(args, "calibration_file")
    if calibration_file:
        calibration_data = load_json_file(calibration_file)
        working_prompts = calibration_data.get("working", [])
        broken_prompts = calibration_data.get("broken", [])
    else:
        working_arg = extract_arg(args, "working")
        broken_arg = extract_arg(args, "broken")
        if working_arg is not None or broken_arg is not None:
            working_prompts = (
                _parse_prompt_list_preserving_trailing_space(working_arg) if working_arg else []
            )
            broken_prompts = (
                _parse_prompt_list_preserving_trailing_space(broken_arg) if broken_arg else []
            )
        else:
            # Use framework-provided calibration datasets (no hardcoded data in CLI)
            calibration = load_calibration_prompts()
            working_prompts = calibration.working
            broken_prompts = calibration.broken

    layer = extract_arg(args, "layer")
    prompts_arg = extract_arg(args, "prompts") or extract_arg(args, "prompt")

    config = UncertaintyConfig(
        model=args.model,
        prompts=_parse_prompt_list_preserving_trailing_space(prompts_arg),
        working_prompts=working_prompts,
        broken_prompts=broken_prompts,
        layer=layer,
        layer_depth_ratio=get_layer_depth_ratio(layer, LayerDepthRatio.DEEP),
        backend=extract_arg(args, "backend"),
        device=extract_arg(args, "device"),
    )

    # Run analysis - all logic is in the service
    result = await UncertaintyService.analyze(config)

    # Print formatted result
    print(result.to_display())


async def introspect_probe(args: Namespace) -> None:
    """Train and evaluate linear probes on model activations.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.probing import ProbeConfig, ProbeService

    # Load probe data from file or CLI args
    probe_file = extract_arg(args, "probe_file")
    if probe_file:
        probe_data = load_json_file(probe_file)
        positive_prompts = probe_data.get("positive", [])
        negative_prompts = probe_data.get("negative", [])
    else:
        # Require explicit data for probing (no defaults)
        positive_arg = extract_arg(args, "positive") or extract_arg(args, "class_a")
        negative_arg = extract_arg(args, "negative") or extract_arg(args, "class_b")

        if not positive_arg or not negative_arg:
            raise ValueError(
                "Probing requires either --probe-file or both class prompt groups"
            )

        positive_prompts = positive_arg.split(Delimiters.PROMPT_SEPARATOR)
        negative_prompts = negative_arg.split(Delimiters.PROMPT_SEPARATOR)

    # Parse layers (using shared utility)
    layer_arg = extract_arg(args, "layer")
    layers_arg = extract_arg(args, "layers")
    layers = [int(layer_arg)] if layer_arg is not None else parse_layers(layers_arg)

    config = ProbeConfig(
        model=args.model,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        positive_label=extract_arg(args, "positive_label")
        or extract_arg(args, "label_a", "Positive"),
        negative_label=extract_arg(args, "negative_label")
        or extract_arg(args, "label_b", "Negative"),
        layers=layers,
        all_layers=extract_arg(args, "all_layers", False),
        ridge_alpha=ProbeDefaults.RIDGE_ALPHA,
        logistic_max_iter=ProbeDefaults.LOGISTIC_MAX_ITER,
        random_seed=AnalysisDefaults.RANDOM_SEED,
        cross_val_folds=AnalysisDefaults.CROSS_VAL_FOLDS,
        backend=extract_arg(args, "backend"),
        device=extract_arg(args, "device"),
    )

    # Run probing - all logic is in the service
    result = await ProbeService.train_and_evaluate(config)

    # Print formatted result
    print(result.to_display())

    # Save if requested
    output_path = extract_arg(args, "output")
    if output_path:
        result.save(output_path)
        print(f"\nResults saved to: {output_path}")


__all__ = [
    "introspect_metacognitive",
    "introspect_probe",
    "introspect_uncertainty",
]
