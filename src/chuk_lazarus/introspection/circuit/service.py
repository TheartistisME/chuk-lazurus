"""Circuit service for CLI commands.

This module provides the CircuitService class that wraps circuit
functionality for CLI commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .._backend_dispatch import backend_matmul, from_backend_tensor, to_backend_tensor


class CircuitCaptureConfig(BaseModel):
    """Configuration for circuit capture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    prompts: list[str] = Field(..., description="Prompts to capture")
    layer: int = Field(..., description="Layer to capture at")
    results: list[int] | None = Field(default=None, description="Expected results")
    extract_direction: bool = Field(default=False, description="Extract direction")
    output_path: str | None = Field(default=None, description="Output path")
    backend: str | None = Field(default=None, description="Optional runtime backend selector")
    device: str | None = Field(default=None, description="Optional device override")


class CircuitCaptureResult(BaseModel):
    """Result of circuit capture."""

    model_config = ConfigDict(frozen=True)

    num_prompts: int = Field(default=0)
    layer: int = Field(default=0)
    output_path: str | None = Field(default=None)
    direction_norm: float | None = Field(default=None)
    activations_shape: list[int] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT CAPTURE",
            f"{'=' * 70}",
            f"\nCaptured {self.num_prompts} prompts at layer {self.layer}",
        ]
        if self.activations_shape:
            lines.append(f"Activations shape: {self.activations_shape}")
        if self.direction_norm is not None:
            lines.append(f"Direction norm: {self.direction_norm:.4f}")
        if self.output_path:
            lines.append(f"Saved to: {self.output_path}")
        return "\n".join(lines)


class CircuitInvokeConfig(BaseModel):
    """Configuration for circuit invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    circuit_file: str = Field(..., description="Circuit file path")
    prompts: list[str] = Field(..., description="Prompts to invoke on")
    method: str = Field(default="project", description="Invocation method")
    coefficient: float | None = Field(default=None, description="Coefficient")
    layer: int | None = Field(default=None, description="Target layer")
    top_k: int = Field(default=10, description="Top-k predictions")
    backend: str | None = Field(default=None, description="Optional runtime backend selector")
    device: str | None = Field(default=None, description="Optional device override")


class CircuitInvokeResult(BaseModel):
    """Result of circuit invocation."""

    model_config = ConfigDict(frozen=True)

    results: list[dict[str, Any]] = Field(default_factory=list)
    method: str = Field(default="")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT INVOCATION",
            f"{'=' * 70}",
            f"Method: {self.method}",
            "",
        ]
        for r in self.results:
            prompt = r.get("prompt", "")[:30]
            prediction = r.get("prediction", "?")
            score = r.get("score", 0)
            lines.append(f"  {prompt:<30} -> {prediction} (score={score:.3f})")
        return "\n".join(lines)


class CircuitTestConfig(BaseModel):
    """Configuration for circuit testing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    circuit_file: str = Field(..., description="Circuit file path")
    prompts: list[str] = Field(..., description="Test prompts")
    expected_results: list[int] | None = Field(default=None, description="Expected results")
    threshold: float = Field(default=0.1, description="Threshold")
    backend: str | None = Field(default=None, description="Optional runtime backend selector")
    device: str | None = Field(default=None, description="Optional device override")


class CircuitTestResult(BaseModel):
    """Result of circuit testing."""

    model_config = ConfigDict(frozen=True)

    accuracy: float = Field(default=0.0)
    results: list[dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    correct: int = Field(default=0)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT TEST",
            f"{'=' * 70}",
            f"\nAccuracy: {self.accuracy:.1%} ({self.correct}/{self.total})",
        ]
        return "\n".join(lines)


class CircuitViewConfig(BaseModel):
    """Configuration for circuit viewing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_file: str = Field(..., description="Circuit file path")
    show_activations: bool = Field(default=False, description="Show activations")
    show_direction: bool = Field(default=True, description="Show direction")


class CircuitViewResult(BaseModel):
    """Result of circuit viewing."""

    model_config = ConfigDict(frozen=True)

    info: dict[str, Any] = Field(default_factory=dict)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT VIEW",
            f"{'=' * 70}",
        ]
        for key, value in self.info.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class CircuitCompareConfig(BaseModel):
    """Configuration for circuit comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_file_a: str = Field(..., description="First circuit file")
    circuit_file_b: str = Field(..., description="Second circuit file")


class CircuitCompareResult(BaseModel):
    """Result of circuit comparison."""

    model_config = ConfigDict(frozen=True)

    similarity: float = Field(default=0.0)
    differences: list[str] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT COMPARISON",
            f"{'=' * 70}",
            f"\nCosine similarity: {self.similarity:.4f}",
        ]
        if self.differences:
            lines.append("\nDifferences:")
            for diff in self.differences:
                lines.append(f"  - {diff}")
        return "\n".join(lines)


class CircuitDecodeConfig(BaseModel):
    """Configuration for circuit decoding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    circuit_file: str = Field(..., description="Circuit file path")
    top_k: int = Field(default=20, description="Top-k tokens")
    activation_index: int = Field(default=0, description="Activation index when no direction exists")
    backend: str | None = Field(default=None, description="Optional runtime backend selector")
    device: str | None = Field(default=None, description="Optional device override")


class CircuitDecodeResult(BaseModel):
    """Result of circuit decoding."""

    model_config = ConfigDict(frozen=True)

    top_tokens: list[dict[str, Any]] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT DECODE",
            f"{'=' * 70}",
        ]
        for token in self.top_tokens[:20]:
            lines.append(f"  {token.get('token', '?')!r}: {token.get('score', 0):.4f}")
        return "\n".join(lines)


class CircuitExportConfig(BaseModel):
    """Configuration for circuit export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_file: str = Field(..., description="Circuit file path")
    output_path: str | None = Field(default=None, description="Output path")
    output_format: str = Field(default="json", description="Output format")
    direction: str = Field(default="TB", description="Graph direction")


class CircuitExportResult(BaseModel):
    """Result of circuit export."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(default="")
    format: str = Field(default="")
    output_path: str | None = Field(default=None)

    def to_display(self) -> str:
        """Format result for display."""
        if self.output_path:
            return f"Exported to: {self.output_path}"
        return self.content


class CircuitService:
    """Service class for circuit operations."""

    @staticmethod
    def _get_embed(model: Any) -> Any:
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        return getattr(model, "embed_tokens", None)

    @staticmethod
    def _get_norm(model: Any) -> Any | None:
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        return getattr(model, "norm", None)

    @classmethod
    def _get_lm_head(cls, model: Any) -> Any | None:
        if hasattr(model, "lm_head") and model.lm_head is not None:
            return model.lm_head

        embed = cls._get_embed(model)
        if embed is not None and hasattr(embed, "as_linear"):
            return embed.as_linear

        return None

    @staticmethod
    def _unwrap_logits(outputs: Any) -> Any:
        return outputs.logits if hasattr(outputs, "logits") else outputs

    @staticmethod
    def _get_embedding_weight(embed: Any) -> Any:
        weight = getattr(embed, "weight", None)
        if weight is None:
            raise AttributeError("Cannot resolve embedding projection weight.")
        return getattr(weight, "weight", weight)

    @staticmethod
    def _reshape_hidden_for_head(hidden: Any, backend_name: str) -> Any:
        if backend_name == "torch":
            hidden = hidden.clone()
            if hidden.ndim == 1:
                return hidden.unsqueeze(0).unsqueeze(0)
            if hidden.ndim == 2:
                return hidden.unsqueeze(1)
            return hidden

        if backend_name == "mlx":
            import mlx.core as mx

            if hidden.ndim == 1:
                return hidden[None, None, :]
            if hidden.ndim == 2:
                return hidden[:, None, :]
            return hidden

        raise NotImplementedError(f"Unsupported backend: {backend_name}")

    @classmethod
    def _project_logits(cls, model: Any, hidden: np.ndarray, backend: Any) -> np.ndarray:
        backend_name = str(backend.name).lower()
        hidden_tensor = to_backend_tensor(hidden, backend=backend)
        hidden_3d = cls._reshape_hidden_for_head(hidden_tensor, backend_name)
        norm = cls._get_norm(model)
        if norm is not None:
            hidden_3d = norm(hidden_3d)

        lm_head = cls._get_lm_head(model)
        if lm_head is not None:
            logits = cls._unwrap_logits(lm_head(hidden_3d))
        else:
            embed = cls._get_embed(model)
            if embed is None:
                raise AttributeError("Cannot resolve embedding or lm_head for vocabulary projection.")
            embed_weight = cls._get_embedding_weight(embed)
            logits = backend_matmul(hidden_3d, embed_weight.T, backend=backend)

        return from_backend_tensor(logits, backend=backend)[0, -1, :]

    @staticmethod
    def _load_pipeline(model_id: str, *, backend: str | None = None, device: str | None = None):
        from ...inference import UnifiedPipeline, UnifiedPipelineConfig
        from ...models_v2.core.backend import get_backend

        resolved_backend = get_backend(name=backend, device=device)
        backend_name = str(resolved_backend.name).lower()
        resolved_device = (
            getattr(resolved_backend, "device", None) if backend_name == "torch" else None
        )

        try:
            pipeline = UnifiedPipeline.from_pretrained(
                model_id,
                pipeline_config=UnifiedPipelineConfig(
                    backend_name=backend_name,
                    device=resolved_device,
                ),
                verbose=False,
            )
        except ValueError as exc:
            message = str(exc).lower()
            if "detect model family" in message or "supported yet" in message:
                raise ValueError(f"Unsupported model: {model_id}") from exc
            raise

        return pipeline, resolved_backend

    @staticmethod
    def _extract_hidden_vector(runtime: Any, backend: Any, prompt: str, *, layer: int) -> np.ndarray:
        residual_state = runtime.extract_residual_state(prompt, layer_index=layer)
        array = from_backend_tensor(residual_state.tensor, backend=backend)
        if array.ndim > 1:
            array = array[0]
        return np.asarray(array, dtype=np.float32).reshape(-1)

    @staticmethod
    def _build_circuit_payload_from_npz(path: str) -> dict[str, Any]:
        from ..models.circuit import CapturedCircuit

        circuit = CapturedCircuit.load(path)
        activations = circuit.activations
        if activations is None:
            entry_activations = [entry.activation for entry in circuit.entries if entry.activation is not None]
            if entry_activations:
                activations = np.stack(entry_activations).astype(np.float32, copy=False)

        payload: dict[str, Any] = {
            "model": circuit.model_id,
            "layer": circuit.layer,
            "num_prompts": circuit.num_entries,
            "prompts": [entry.prompt for entry in circuit.entries],
        }

        results = [entry.result for entry in circuit.entries]
        if any(result is not None for result in results):
            payload["results"] = results
        if activations is not None:
            payload["activations"] = activations.tolist()
        if circuit.direction is not None:
            payload["direction"] = np.asarray(
                circuit.direction.direction, dtype=np.float32
            ).tolist()

        return payload

    @classmethod
    def _load_circuit_data(cls, circuit_file: str) -> dict[str, Any]:
        path = Path(circuit_file)
        if path.suffix.lower() == ".npz":
            try:
                return cls._build_circuit_payload_from_npz(circuit_file)
            except Exception:
                pass

        with open(circuit_file) as f:
            return json.load(f)

    @classmethod
    def _save_circuit_data(
        cls,
        *,
        config: CircuitCaptureConfig,
        activations: np.ndarray,
        direction: np.ndarray | None,
    ) -> None:
        if config.output_path is None:
            return

        path = Path(config.output_path)
        if path.suffix.lower() == ".npz":
            from ..models.circuit import CapturedCircuit, CircuitDirection, CircuitEntry

            entries = [
                CircuitEntry(
                    prompt=prompt,
                    result=config.results[idx] if config.results else None,
                    activation=activations[idx],
                )
                for idx, prompt in enumerate(config.prompts)
            ]
            circuit_direction = None
            if direction is not None:
                vector = np.asarray(direction, dtype=np.float32)
                circuit_direction = CircuitDirection(
                    direction=vector,
                    norm=float(np.linalg.norm(vector)),
                )

            CapturedCircuit(
                model_id=config.model,
                layer=config.layer,
                entries=entries,
                direction=circuit_direction,
                activations=activations,
            ).save(config.output_path)
            return

        output_data: dict[str, Any] = {
            "model": config.model,
            "layer": config.layer,
            "num_prompts": len(config.prompts),
            "prompts": config.prompts,
            "activations": activations.tolist(),
        }
        if config.results:
            output_data["results"] = config.results
        if direction is not None:
            output_data["direction"] = np.asarray(direction, dtype=np.float32).tolist()

        with open(config.output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    @staticmethod
    def _resolve_decode_vector(
        circuit_data: dict[str, Any],
        *,
        activation_index: int,
    ) -> np.ndarray:
        direction = np.asarray(circuit_data.get("direction", []), dtype=np.float32).reshape(-1)
        if direction.size:
            return direction

        activations = np.asarray(circuit_data.get("activations", []), dtype=np.float32)
        if activations.ndim == 1 and activations.size:
            return activations.reshape(-1)
        if activations.ndim >= 2 and activations.shape[0] > 0:
            index = max(0, min(int(activation_index), activations.shape[0] - 1))
            return activations[index].reshape(-1)

        return np.array([], dtype=np.float32)

    @classmethod
    async def capture(cls, config: CircuitCaptureConfig) -> CircuitCaptureResult:
        """Capture circuit activations."""
        pipeline, backend = cls._load_pipeline(
            config.model,
            backend=config.backend,
            device=config.device,
        )

        # Collect activations
        activations = np.stack(
            [
                cls._extract_hidden_vector(pipeline.runtime, backend, prompt, layer=config.layer)
                for prompt in config.prompts
            ]
        ).astype(np.float32, copy=False)

        # Extract direction if requested
        direction = None
        direction_norm = None
        if config.extract_direction and config.results:
            # Use Ridge regression to find direction
            from sklearn.linear_model import Ridge

            y = np.array(config.results)
            ridge = Ridge(alpha=1.0)
            ridge.fit(activations, y)
            direction = ridge.coef_
            direction_norm = float(np.linalg.norm(direction))

        cls._save_circuit_data(
            config=config,
            activations=activations,
            direction=None if direction is None else np.asarray(direction, dtype=np.float32),
        )

        return CircuitCaptureResult(
            num_prompts=len(config.prompts),
            layer=config.layer,
            output_path=config.output_path,
            direction_norm=direction_norm,
            activations_shape=list(activations.shape),
        )

    @classmethod
    async def invoke(cls, config: CircuitInvokeConfig) -> CircuitInvokeResult:
        """Invoke a captured circuit."""
        circuit_data = cls._load_circuit_data(config.circuit_file)
        direction = np.asarray(circuit_data.get("direction", []), dtype=np.float32).reshape(-1)
        layer = config.layer or circuit_data.get("layer", 0)
        pipeline, backend = cls._load_pipeline(
            config.model,
            backend=config.backend,
            device=config.device,
        )

        results = []
        for prompt in config.prompts:
            h = cls._extract_hidden_vector(pipeline.runtime, backend, prompt, layer=layer)

            # Project onto direction
            if direction.size:
                score = float(np.dot(h, direction) / (np.linalg.norm(direction) + 1e-8))
                prediction = "positive" if score > 0 else "negative"
            else:
                score = 0.0
                prediction = "unknown"

            results.append(
                {
                    "prompt": prompt,
                    "score": score,
                    "prediction": prediction,
                }
            )

        return CircuitInvokeResult(
            results=results,
            method=config.method.value if hasattr(config.method, "value") else str(config.method),
        )

    @classmethod
    async def test(cls, config: CircuitTestConfig) -> CircuitTestResult:
        """Test circuit predictions."""
        # Use invoke to get predictions
        invoke_config = CircuitInvokeConfig(
            model=config.model,
            circuit_file=config.circuit_file,
            prompts=config.prompts,
            backend=config.backend,
            device=config.device,
        )
        invoke_result = await cls.invoke(invoke_config)

        # Compare with expected
        correct = 0
        total = len(config.prompts)
        results = []

        expected = config.expected_results or [0] * total

        for i, (r, exp) in enumerate(zip(invoke_result.results, expected)):
            pred = 1 if r["score"] > config.threshold else 0
            is_correct = pred == exp
            if is_correct:
                correct += 1

            results.append(
                {
                    **r,
                    "expected": exp,
                    "predicted": pred,
                    "correct": is_correct,
                }
            )

        return CircuitTestResult(
            accuracy=correct / total if total > 0 else 0.0,
            results=results,
            total=total,
            correct=correct,
        )

    @classmethod
    async def view(cls, config: CircuitViewConfig) -> CircuitViewResult:
        """View circuit contents."""
        circuit_data = cls._load_circuit_data(config.circuit_file)

        info = {
            "model": circuit_data.get("model", "unknown"),
            "layer": circuit_data.get("layer", "unknown"),
            "num_prompts": circuit_data.get(
                "num_prompts",
                len(circuit_data.get("prompts", [])),
            ),
        }

        if config.show_direction and "direction" in circuit_data:
            direction = np.asarray(circuit_data["direction"], dtype=np.float32)
            info["direction_dim"] = len(direction)
            info["direction_norm"] = float(np.linalg.norm(direction))

        return CircuitViewResult(info=info)

    @classmethod
    async def compare(cls, config: CircuitCompareConfig) -> CircuitCompareResult:
        """Compare two circuits."""
        circuit_a = cls._load_circuit_data(config.circuit_file_a)
        circuit_b = cls._load_circuit_data(config.circuit_file_b)

        differences = []

        # Compare metadata
        if circuit_a.get("model") != circuit_b.get("model"):
            differences.append(
                f"Different models: {circuit_a.get('model')} vs {circuit_b.get('model')}"
            )
        if circuit_a.get("layer") != circuit_b.get("layer"):
            differences.append(
                f"Different layers: {circuit_a.get('layer')} vs {circuit_b.get('layer')}"
            )

        # Compare directions
        similarity = 0.0
        dir_a = np.asarray(circuit_a.get("direction", []), dtype=np.float32)
        dir_b = np.asarray(circuit_b.get("direction", []), dtype=np.float32)

        if len(dir_a) > 0 and len(dir_b) > 0 and len(dir_a) == len(dir_b):
            similarity = float(
                np.dot(dir_a, dir_b) / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b) + 1e-8)
            )
        elif len(dir_a) != len(dir_b):
            differences.append(f"Different direction dimensions: {len(dir_a)} vs {len(dir_b)}")

        return CircuitCompareResult(
            similarity=similarity,
            differences=differences,
        )

    @classmethod
    async def decode(cls, config: CircuitDecodeConfig) -> CircuitDecodeResult:
        """Decode circuit through vocabulary."""
        circuit_data = cls._load_circuit_data(config.circuit_file)
        vector = cls._resolve_decode_vector(
            circuit_data,
            activation_index=config.activation_index,
        )
        if vector.size == 0:
            return CircuitDecodeResult(top_tokens=[])

        pipeline, backend = cls._load_pipeline(
            config.model,
            backend=config.backend,
            device=config.device,
        )
        scores = cls._project_logits(pipeline.model, vector, backend)
        top_indices = np.argsort(scores)[-config.top_k :][::-1]

        top_tokens = []
        for idx in top_indices:
            token = pipeline.tokenizer.decode([int(idx)])
            top_tokens.append(
                {
                    "token": token,
                    "token_id": int(idx),
                    "score": float(scores[idx]),
                }
            )

        return CircuitDecodeResult(top_tokens=top_tokens)

    @classmethod
    async def export(cls, config: CircuitExportConfig) -> CircuitExportResult:
        """Export circuit in various formats."""
        circuit_data = cls._load_circuit_data(config.circuit_file)

        if config.output_format == "json":
            content = json.dumps(circuit_data, indent=2)
        elif config.output_format == "dot":
            # Simple DOT format
            content = cls._to_dot(circuit_data, config.direction)
        elif config.output_format == "mermaid":
            content = cls._to_mermaid(circuit_data, config.direction)
        else:
            content = json.dumps(circuit_data, indent=2)

        if config.output_path:
            with open(config.output_path, "w") as f:
                f.write(content)

        return CircuitExportResult(
            content=content,
            format=config.output_format,
            output_path=config.output_path,
        )

    @staticmethod
    def _to_dot(circuit_data: dict, direction: str = "TB") -> str:
        """Convert circuit to DOT format."""
        lines = [
            "digraph Circuit {",
            f"  rankdir={direction};",
            f'  label="Circuit at layer {circuit_data.get("layer", "?")}";',
            "  node [shape=box];",
        ]

        # Add nodes for prompts
        for i, prompt in enumerate(circuit_data.get("prompts", [])[:10]):
            short = prompt[:20].replace('"', '\\"')
            lines.append(f'  p{i} [label="{short}..."];')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _to_mermaid(circuit_data: dict, direction: str = "TB") -> str:
        """Convert circuit to Mermaid format."""
        lines = [
            f"graph {direction}",
            f"  subgraph Layer{circuit_data.get('layer', '?')}",
        ]

        for i, prompt in enumerate(circuit_data.get("prompts", [])[:10]):
            short = prompt[:20].replace('"', "'")
            lines.append(f'    p{i}["{short}..."]')

        lines.append("  end")
        return "\n".join(lines)


__all__ = [
    "CircuitCaptureConfig",
    "CircuitCaptureResult",
    "CircuitCompareConfig",
    "CircuitCompareResult",
    "CircuitDecodeConfig",
    "CircuitDecodeResult",
    "CircuitExportConfig",
    "CircuitExportResult",
    "CircuitInvokeConfig",
    "CircuitInvokeResult",
    "CircuitService",
    "CircuitTestConfig",
    "CircuitTestResult",
    "CircuitViewConfig",
    "CircuitViewResult",
]
