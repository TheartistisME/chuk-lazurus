"""Build ``exp.yaml`` — minimal experiment config stub."""

from __future__ import annotations

from pathlib import Path


def build(out_dir: Path, *, model: str | None = None, seed: int = 0) -> Path:
    out_path = out_dir / "exp.yaml"
    model_spec = model or "Qwen/Qwen2.5-0.5B-Instruct"
    out_path.write_text(
        "\n".join(
            [
                f"model: {model_spec}",
                f"seed: {seed}",
                "batch_size: 2",
                "max_steps: 4",
                "",
            ]
        )
    )
    return out_path


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
