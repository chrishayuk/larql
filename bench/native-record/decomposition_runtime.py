"""Independent E/C/A interventions on actual FFN feature activations."""
import numpy as np

from binding_runtime import Runtime


def expected_sequence(raw, code, selected, amplitude):
    """NumPy specification in descending slot order, separate from GPU masks."""
    out = raw.copy()
    if code[0] == '1':
        out[:, :-1, :] = 0
    if code[1] == '1':
        for index in range(out.shape[-1]):
            if index != selected:
                out[0, -1, index] = 0
    if code[2] == '1':
        out[0, -1, selected] = amplitude
    return out


class DecompositionRuntime(Runtime):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.slots = list(range(self.G.shape[0] - 1, self.G.shape[0] - 13, -1))
        self.code, self.selected, self.amplitude = '000', 0, 0.
        self.extra = {}
        parent_linear = self.linear_cls.__call__
        norm = self.model.model.layers[26].post_feedforward_layernorm
        self.norm_cls, self.norm_call = type(norm), type(norm).__call__

        def linear(module, x):
            if module is self.mlp.down_proj:
                original = x
                if self.code != '000':
                    shape = (*x.shape[:-1], 12)
                    mask, values = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float32)
                    column = self.slots[self.selected] - min(self.slots)
                    if self.code[0] == '1':
                        mask[:, :-1, :] = True
                    if self.code[1] == '1':
                        mask[0, -1, :] = True
                        mask[0, -1, column] = False
                    if self.code[2] == '1':
                        mask[0, -1, column] = True
                        values[0, -1, column] = self.amplitude
                    tail = self.mx.where(self.mx.array(mask), self.mx.array(values).astype(x.dtype),
                                         x[..., min(self.slots):])
                    x = self.mx.concatenate([x[..., :min(self.slots)], tail], axis=-1)
                self.extra['raw_sequence'] = original[..., self.mx.array(self.slots)]
                self.extra['edited_sequence'] = x[..., self.mx.array(self.slots)]
                self.extra['native_max_delta'] = self.mx.max(self.mx.abs(
                    original[..., :min(self.slots)].astype(self.mx.float32) -
                    x[..., :min(self.slots)].astype(self.mx.float32)))
            return parent_linear(module, x)

        def normalization(module, x):
            out = self.norm_call(module, x)
            if module is norm:
                self.extra['pre_norm'], self.extra['post_norm'] = x[0, -1], out[0, -1]
            return out

        self.linear_cls.__call__, self.norm_cls.__call__ = linear, normalization

    def intervention_forward(self, prompt, code, selected, amplitude):
        assert code in [f'{i:03b}' for i in range(8)]
        self.code, self.selected, self.amplitude = code, selected, amplitude
        self.extra.clear()
        ll, cap = super().forward(prompt)
        extra = {k: np.array(v.astype(self.mx.float32)) for k, v in self.extra.items()}
        assert float(extra['native_max_delta']) == 0.
        np.testing.assert_array_equal(extra['edited_sequence'],
                                      expected_sequence(extra['raw_sequence'], code, selected, amplitude))
        return ll, cap, extra

    def close(self):
        self.norm_cls.__call__ = self.norm_call
        super().close()
