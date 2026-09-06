"""Native row replacement; an explicitly separate C+A positive control."""
import numpy as np
from binding_runtime import Runtime


class ControllerRuntime(Runtime):
    def __init__(self, model_path):
        super().__init__(model_path)
        import mlx.nn as nn
        self.activation_fn = nn.gelu_approx
        self.slots = list(range(self.G.shape[0] - 1, self.G.shape[0] - 13, -1))
        from mlx.utils import tree_flatten
        self.flatten = tree_flatten
        self.parameter_refs = dict(tree_flatten(self.model.parameters()))
        self.edit_names = {name for name, value in self.parameter_refs.items()
                           if any(value is w for w in self.original)}
        assert len(self.edit_names) == 3
        self.oracle = False
        self.record = None
        self.amplitude = 0.
        self.sequence = {}
        parent = self.linear_cls.__call__

        def linear(module, x):
            if module is self.mlp.down_proj:
                self.sequence['raw'] = x[..., self.mx.array(self.slots)]
                if self.oracle:
                    last = np.zeros((1, 1, 12), dtype=np.float32)
                    if self.record is not None:
                        last[0, 0, self.slots[self.record] - min(self.slots)] = self.amplitude
                    tail = self.mx.concatenate([x[:, :-1, min(self.slots):],
                        self.mx.array(last).astype(x.dtype)], axis=1)
                    x = self.mx.concatenate([x[..., :min(self.slots)], tail], axis=-1)
                self.sequence['actual'] = x[..., self.mx.array(self.slots)]
            return parent(module, x)

        self.linear_cls.__call__ = linear

    def checked_forward(self, prompt, oracle=False, record=None, amplitude=0.):
        self.oracle, self.record, self.amplitude = oracle, record, amplitude
        self.sequence.clear()
        logits, cap = self.forward(prompt)
        seq = {k: np.array(v.astype(self.mx.float32)) for k, v in self.sequence.items()}
        expected = seq['raw'].copy()
        if oracle:
            expected[:, -1, :] = 0
            if record is not None:
                expected[0, -1, record] = amplitude
        np.testing.assert_array_equal(seq['actual'], expected)
        return logits, cap, seq

    def fitted_weights(self, existing, rows):
        weights = []
        for original, fitted in zip(existing[:2], rows):
            full = np.array(original.astype(self.mx.float32))
            full[self.slots] = fitted
            new = self.mx.array(full).astype(original.dtype)
            np.testing.assert_array_equal(np.array(new[:min(self.slots)].astype(self.mx.float32)),
                                          np.array(original[:min(self.slots)].astype(self.mx.float32)))
            weights.append(new)
        return weights[0], weights[1], existing[2]

    def frozen_check(self):
        current = dict(self.flatten(self.model.parameters()))
        assert current.keys() == self.parameter_refs.keys()
        assert all(current[k] is v for k, v in self.parameter_refs.items() if k not in self.edit_names)
