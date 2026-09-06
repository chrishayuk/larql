"""Original editor plus read-only hooks on actual MLP operations."""
import importlib.util
import os

import numpy as np

from run_address import HERE, DECOYS
from binding_fixture import canonical, TRAIN


class Runtime:
    def __init__(self, model_path, layers=(26,)):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import mlx.core as mx
        from chuk_lazarus.models_v2.loader import load_model, ModelDType
        self.mx = mx
        spec = importlib.util.spec_from_file_location('original_native', HERE / 'vendor/native.py')
        self.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.native)
        loaded = load_model(str(model_path), dtype=ModelDType.BFLOAT16)
        self.model, self.tok = loaded.model, loaded.tokenizer
        self.model.eval()
        assert len(self.model.model.layers) == 34
        self.layers = layers
        self.mlp = self.model.model.layers[26].mlp
        self.original = (self.mlp.gate_proj.weight, self.mlp.up_proj.weight, self.mlp.down_proj.weight)
        self.G, self.U, self.D = [np.array(w.astype(mx.float32)) for w in self.original]
        self.scales = [float(np.median(np.linalg.norm(a, axis=axis)))
                       for a, axis in [(self.G, 1), (self.U, 1), (self.D, 0)]]
        self.capture = {}
        self.mlp_cls, self.mlp_call = type(self.mlp), type(self.mlp).__call__
        self.linear_cls, self.linear_call = type(self.mlp.gate_proj), type(self.mlp.gate_proj).__call__
        mlps, linears = {}, {}
        for layer in layers:
            mlp = self.model.model.layers[layer].mlp
            mlps[id(mlp)] = layer
            for kind in ['gate', 'up', 'down']:
                linear = getattr(mlp, kind + '_proj')
                assert type(linear) is self.linear_cls
                linears[id(linear)] = (layer, kind)

        def mlp_hook(module, x):
            if id(module) in mlps:
                self.capture.setdefault(mlps[id(module)], {})['x'] = x[0, -1]
            return self.mlp_call(module, x)

        def linear_hook(module, x):
            output = self.linear_call(module, x)
            if id(module) in linears:
                layer, kind = linears[id(module)]
                cap = self.capture.setdefault(layer, {})
                cap[kind] = output[0, -1]
                if kind == 'down':
                    cap['activation'] = x[0, -1]
            return output

        self.mlp_cls.__call__, self.linear_cls.__call__ = mlp_hook, linear_hook

    def close(self):
        self.assign(self.original)
        self.mlp_cls.__call__, self.linear_cls.__call__ = self.mlp_call, self.linear_call

    def assign(self, weights):
        self.mlp.gate_proj.weight, self.mlp.up_proj.weight, self.mlp.down_proj.weight = weights

    def forward(self, prompt):
        self.capture.clear()
        ll = self.model(self.mx.array([self.tok.encode(prompt)])).logits[0, -1]
        self.mx.eval(ll)
        cap = {layer: {k: np.array(v.astype(self.mx.float32)) for k, v in values.items()}
               for layer, values in self.capture.items()}
        return np.array(ll.astype(self.mx.float32)), cap

    def token(self, value):
        ids = [i for i in self.tok.encode(' ' + value) if i != self.tok.bos_token_id]
        if len(ids) != 1:
            raise ValueError(f'Expected a single token for {value!r}: {ids}')
        return ids[0]

    def value(self, value):
        return self.native.unit(np.array(self.model.model.embed_tokens.weight[self.token(value)]
                                         .astype(self.mx.float32)))

    def prepare(self, facts):
        self.assign(self.original)
        self.facts = facts
        self.addresses = [self.forward(canonical(e, r))[1][26]['x'] for e, r, _ in facts]
        self.decoys = [self.forward(p)[1][26]['x'] for p in DECOYS]
        self.negative_forms = {}
        for i, (e, r, _) in enumerate(facts):
            self.negative_forms[i] = [self.forward(form.format(e=e, r=r))[1][26]['x'] for form in TRAIN]
        self.slots = list(range(self.G.shape[0] - 1, self.G.shape[0] - len(facts) - 1, -1))

    def make_edit(self, explicit=False):
        G, U, D = self.G.copy(), self.U.copy(), self.D.copy()
        keys = []
        for i, (e, r, value) in enumerate(self.facts):
            others = self.decoys + [a for j, a in enumerate(self.addresses) if j != i]
            if explicit:
                for j, (other_e, other_r, _) in enumerate(self.facts):
                    if j != i and (other_e == e or other_r == r):
                        others += self.negative_forms[j]
            key = self.native.unit(self.native.unique_part(self.addresses[i], others))
            keys.append(key)
            slot = self.slots[i]
            G[slot] = key * self.scales[0] * 30.
            U[slot] = key * self.scales[1]
            D[:, slot] = self.value(value) * self.scales[2] * .1
        weights = tuple(self.mx.array(a).astype(w.dtype) for a, w in zip([G, U, D], self.original))
        return weights, np.stack(keys)

    def replace(self, weights, fact, value):
        down = np.array(weights[2].astype(self.mx.float32))
        down[:, self.slots[fact]] = self.value(value) * self.scales[2] * .1
        return weights[0], weights[1], self.mx.array(down).astype(weights[2].dtype)

    def slot_metrics(self, cap):
        layer = cap[26]
        d = np.array(self.mlp.down_proj.weight[:, self.mx.array(self.slots)].astype(self.mx.float32))
        activation = layer['activation'][self.slots]
        return dict(gate=layer['gate'][self.slots].tolist(), up=layer['up'][self.slots].tolist(),
                    activation=activation.tolist(), contribution_norm=(np.abs(activation) *
                        np.linalg.norm(d, axis=0)).tolist(), mlp_output_norm=float(np.linalg.norm(layer['down'])))
