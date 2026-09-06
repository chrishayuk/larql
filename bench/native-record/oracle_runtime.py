"""Artificial routing at the true down-projection input; real norm and downstream."""
import numpy as np

from binding_runtime import Runtime


class OracleRuntime(Runtime):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.slots = list(range(self.G.shape[0] - 1, self.G.shape[0] - 13, -1))
        self.mode, self.selected, self.amplitude = 'actual', None, 0.
        self.extra = {}
        parent_linear = self.linear_cls.__call__
        norm = self.model.model.layers[26].post_feedforward_layernorm
        self.norm_cls, self.norm_call = type(norm), type(norm).__call__

        def linear(module, x):
            if module is self.mlp.down_proj:
                original = x
                if self.mode != 'actual':
                    tail = np.zeros((*x.shape[:-1], len(self.slots)), dtype=np.float32)
                    if self.selected is not None:
                        tail[0, -1, self.slots[self.selected] - min(self.slots)] = self.amplitude
                    x = self.mx.concatenate([x[..., :min(self.slots)],
                                             self.mx.array(tail).astype(x.dtype)], axis=-1)
                self.extra['raw_activation'] = original[0, -1, self.mx.array(self.slots)]
                self.extra['native_max_delta'] = self.mx.max(self.mx.abs(
                    original[..., :min(self.slots)].astype(self.mx.float32) -
                    x[..., :min(self.slots)].astype(self.mx.float32)))
                self.extra['edited_sequence'] = x[..., self.mx.array(self.slots)]
            return parent_linear(module, x)

        def normalization(module, x):
            result = self.norm_call(module, x)
            if module is norm:
                self.extra['pre_norm'] = x[0, -1]
                self.extra['post_norm'] = result[0, -1]
            return result

        self.linear_cls.__call__, self.norm_cls.__call__ = linear, normalization

    def close(self):
        self.norm_cls.__call__ = self.norm_call
        super().close()

    def routed_forward(self, prompt, mode='actual', selected=None, amplitude=0.):
        self.mode, self.selected, self.amplitude = mode, selected, amplitude
        self.extra.clear()
        ll, cap = super().forward(prompt)
        extra = {k: np.array(v.astype(self.mx.float32)) for k, v in self.extra.items()}
        assert float(extra['native_max_delta']) == 0.
        if mode != 'actual':
            expected = np.zeros_like(extra['edited_sequence'])
            if selected is not None:
                expected[0, -1, selected] = amplitude
            np.testing.assert_array_equal(extra['edited_sequence'], expected)
        return ll, cap, extra

    def scale_column(self, weights, fact, scale):
        down = np.array(weights[2].astype(self.mx.float32))
        down[:, self.slots[fact]] *= scale
        return weights[0], weights[1], self.mx.array(down).astype(weights[2].dtype)


def vector_metrics(pre, post, zero_pre, zero_post, contribution):
    pre, post, zero_pre, zero_post = [np.asarray(x, np.float64)
                                    for x in [pre, post, zero_pre, zero_post]]
    dp, dn = pre - zero_pre, post - zero_post
    pn, nn = float(np.linalg.norm(dp)), float(np.linalg.norm(dn))
    whole = float(np.linalg.norm(pre))
    return dict(ffn_norm=whole, post_norm_norm=float(np.linalg.norm(post)),
                contribution_norm=float(contribution),
                contribution_over_ffn=float(contribution / whole) if whole else None,
                pre_norm_delta=pn, post_norm_delta=nn,
                normalization_delta_gain=nn / pn if pn else None,
                delta_direction_cosine=float(dp @ dn / (pn * nn)) if pn and nn else None)
