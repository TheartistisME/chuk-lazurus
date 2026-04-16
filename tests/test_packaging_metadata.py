from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def _array_body(name: str, text: str) -> str:
    match = re.search(rf"^{re.escape(name)}\s*=\s*\[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert match is not None, f"Could not find array for {name!r} in pyproject.toml"
    return match.group(1)


def test_cuda_optional_dependencies_are_resolvable_shape():
    text = PYPROJECT.read_text()

    assert '"huggingface-hub>=0.34.0,<1.0.0"' in text

    cuda_block = _array_body("cuda", text)
    assert '"torch>=2.9.0"' in cuda_block
    assert '"safetensors>=0.4.0"' in cuda_block
    assert '"vllm>=0.11.1"' not in cuda_block
    assert '"flash-attn>=2.8.0"' not in cuda_block

    cuda_accel_block = _array_body("cuda-accel", text)
    assert '"vllm>=0.11.1 ; sys_platform == \'linux\'"' in cuda_accel_block
    assert '"flash-attn>=2.8.0 ; sys_platform == \'linux\'"' in cuda_accel_block

    torch_cuda_block = _array_body("torch-cuda", text)
    assert '"chuk-lazarus[cuda]"' in torch_cuda_block

    all_block = _array_body("all", text)
    assert '"chuk-lazarus[circuit,openai,server]"' in all_block
    assert '"chuk-lazarus[cuda] ; sys_platform == \'linux\'"' in all_block
    assert (
        '"chuk-lazarus[mlx,fast] ; sys_platform == \'darwin\' and platform_machine == \'arm64\'"'
        in all_block
    )
    assert '"chuk-lazarus[torch] ; sys_platform == \'win32\'"' in all_block
