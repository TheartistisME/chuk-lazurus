"""Cross-framework model adapter (EWS-9).

Lazy-import-clean: torch/mlx imports are function-scope so this module can
be imported under ``CHUK_BACKEND=torch`` without pulling in mlx.
"""

from __future__ import annotations

from typing import Any


class ModelAdapter:
    def __init__(self, framework: str = "mlx", model: Any = None):
        self.framework = framework
        self.model = model

    def to_tensor(self, data):
        if self.framework == "torch":
            import torch

            return torch.tensor(data)
        import mlx.core as mx

        return mx.array(data)

    def forward(self, input_tensor):
        if self.framework == "torch":
            import torch

            with torch.no_grad():
                return self.model(input_tensor).cpu().numpy()
        return self.model(input_tensor)

    def argmax(self, output, axis: int = -1):
        if self.framework == "torch":
            import torch

            return torch.argmax(torch.as_tensor(output), dim=axis).tolist()
        import mlx.core as mx

        return mx.argmax(mx.array(output), axis=axis).tolist()

    def create_value_and_grad_fn(self, loss_function):
        if self.framework == "torch":

            def value_and_grad(inputs, targets):
                outputs = self.model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                return loss.item(), [p.grad for p in self.model.parameters()]

            return value_and_grad

        import mlx.nn as nn

        return nn.value_and_grad(self.model, loss_function)

    def load_tensor_from_file(self, batch_path):
        if self.framework == "torch":
            import torch

            return torch.load(batch_path)
        import mlx.core as mx

        return mx.load(batch_path)
