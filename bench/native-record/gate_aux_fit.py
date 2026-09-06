"""Previous optimizer with exactly one optional, training-only gate penalty."""
import numpy as np
from controller_fit import CHECKPOINTS, targets, pattern_loss, delivery

STRENGTHS = [.0001, .001, .01]


def auxiliary(mx, gate_scores, target):
    positive = mx.sum(target, axis=1)
    intended = mx.sum(gate_scores * target, axis=1)
    return mx.sum(positive * mx.maximum(1. - intended, 0.) ** 2) / mx.sum(positive)


def fit_aux(mx, activation_fn, X, rows, initial, canonical, config, steps, strength, callback):
    train = np.array([r['split'] == 'train' for r in rows])
    valid = ~train
    Y = targets(rows, canonical)
    x = mx.array(X[train])
    y = mx.array(Y[train])
    positive = Y[train].sum(axis=1) != 0
    row_weight = mx.array(np.where(positive, .5 / positive.sum(), .5 / (~positive).sum()).astype(np.float32))
    scale = [mx.array(np.linalg.norm(w, axis=1, keepdims=True)) for w in initial]
    origin = [mx.array(w) / s for w, s in zip(initial, scale)]
    q = [v for v in origin]
    amp = mx.array(np.asarray(canonical, dtype=np.float32))

    def objective(params):
        g, u = [v * s for v, s in zip(params, scale)]
        gate_scores = x @ g.T
        pred = activation_fn(gate_scores) * (x @ u.T) / amp
        loss = mx.sum(row_weight * mx.sum((pred - y) ** 2, axis=1))
        reg = sum(mx.mean(mx.sum((v - o) ** 2, axis=1)) for v, o in zip(params, origin))
        result = loss + config['regularization'] * reg
        return result + strength * auxiliary(mx, gate_scores, y) if strength else result

    value_grad = mx.value_and_grad(objective)
    moment = [mx.zeros_like(v) for v in q]
    variance = [mx.zeros_like(v) for v in q]
    for step in range(1, steps + 1):
        loss, grad = value_grad(q)
        norm = mx.sqrt(sum(mx.sum(g * g) for g in grad))
        grad = [g / mx.maximum(norm, 1.) for g in grad]
        moment = [.9 * m + .1 * g for m, g in zip(moment, grad)]
        variance = [.999 * v + .001 * g * g for v, g in zip(variance, grad)]
        q = [p - config['lr'] * (m / (1 - .9 ** step)) /
             (mx.sqrt(v / (1 - .999 ** step)) + 1e-8) for p, m, v in zip(q, moment, variance)]
        mx.eval(q, moment, variance, loss)
        if step in CHECKPOINTS or step == steps:
            assert np.isfinite(float(loss))
            weights = [v * s for v, s in zip(q, scale)]
            xb = mx.array(X).astype(mx.bfloat16)
            gb, ub = [w.astype(mx.bfloat16) for w in weights]
            pred = np.array((activation_fn(xb @ gb.T) * (xb @ ub.T)).astype(mx.float32)) / np.asarray(canonical)
            callback(step, [np.array(w.astype(mx.float32)) for w in [gb, ub]], dict(
                objective=float(loss), train_loss=pattern_loss(pred[train], Y[train]),
                validation_loss=pattern_loss(pred[valid], Y[valid]),
                train_delivery=sum(delivery(p * canonical, canonical, r['fact'])['delivery_ok']
                                   for p, r, keep in zip(pred, rows, train) if keep),
                validation_delivery=sum(delivery(p * canonical, canonical, r['fact'])['delivery_ok']
                                        for p, r, keep in zip(pred, rows, valid) if keep)))
