"""Fit only two 12-row matrices to the native activation-delivery objective."""
import numpy as np

CANDIDATES = [dict(lr=lr, regularization=reg) for lr in [0.0003, 0.001]
              for reg in [0.0001, 0.01]]
CHECKPOINTS = [400, 800, 1200, 1600]


def targets(rows, canonical):
    y = np.zeros((len(rows), 12), dtype=np.float32)
    for i, row in enumerate(rows):
        if row['fact'] is not None:
            y[i, row['fact']] = 1.
    return y


def pattern_loss(pred, target):
    error = np.sum((pred - target) ** 2, axis=1)
    positive = np.sum(target, axis=1) != 0
    assert positive.any() and (~positive).any()
    return float(.5 * np.mean(error[positive]) + .5 * np.mean(error[~positive]))


def delivery(activation, canonical, fact):
    ratio = np.asarray(activation) / np.asarray(canonical)
    chosen = int(np.argmax(ratio)) if float(np.max(ratio)) >= .5 else None
    if fact is None:
        return dict(selected=chosen, selection_correct=chosen is None,
                    delivery_ok=bool(np.max(np.abs(ratio)) <= .1),
                    target_ratio=None, competitor_max=float(np.max(np.abs(ratio))))
    competition = np.delete(ratio, fact)
    return dict(selected=chosen, selection_correct=chosen == fact,
                delivery_ok=bool(abs(float(ratio[fact]) - 1.) <= .2 and np.max(np.abs(competition)) <= .1),
                target_ratio=float(ratio[fact]), competitor_max=float(np.max(np.abs(competition))))


def fit(mx, activation_fn, X, rows, initial, canonical, config, steps, callback):
    """No full-model graph or logits: only cached MLP inputs and 61,440 scalars."""
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
        pred = activation_fn(x @ g.T) * (x @ u.T) / amp
        loss = mx.sum(row_weight * mx.sum((pred - y) ** 2, axis=1))
        reg = sum(mx.mean(mx.sum((v - o) ** 2, axis=1)) for v, o in zip(params, origin))
        return loss + config['regularization'] * reg

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
            # Selection uses deployed bf16 arithmetic; training is f32.
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

