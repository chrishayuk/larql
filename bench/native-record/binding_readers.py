"""Small numpy readers; settings fit on development entities only."""
import numpy as np


def unit_rows(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pairs(queries, keys):
    q, k = unit_rows(queries)[:, None, :], unit_rows(keys)[None, :, :]
    return np.concatenate([np.abs(q - k), q * k], axis=-1)


def auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    d = np.asarray(positive)[:, None] - np.asarray(negative)[None, :]
    return float(np.mean((d > 0) + .5 * (d == 0)))


class RidgeMatch:
    def __init__(self, queries, keys, labels, regularization):
        raw = pairs(queries, keys).reshape(-1, 2 * keys.shape[1])
        self.mean, self.std = raw.mean(0), np.maximum(raw.std(0), 1e-6)
        features = self.features(raw)
        y = (np.arange(len(keys))[None, :] == np.asarray(labels)[:, None]).ravel()
        w = np.where(y, len(keys) - 1, 1.).astype(float)
        w /= w.mean()
        root = np.sqrt(w)
        weighted = features * root[:, None]
        dual = np.linalg.solve(weighted @ weighted.T + regularization * np.eye(len(raw)),
                               np.where(y, 1., -1.) * root)
        self.beta = weighted.T @ dual

    def features(self, raw):
        f = (raw - self.mean) / self.std / np.sqrt(raw.shape[-1])
        return np.concatenate([f, np.ones((*f.shape[:-1], 1))], axis=-1)

    def scores(self, queries, keys):
        return self.features(pairs(queries, keys)) @ self.beta


def choose_threshold(known, unknown):
    values = np.unique(np.r_[known, unknown])
    thresholds = np.r_[np.nextafter(values.min(), -np.inf), values,
                       np.nextafter(values.max(), np.inf)]
    # Larger threshold wins ties; no test examples are used.
    return float(max(thresholds, key=lambda t: (
        (np.mean(known >= t) + np.mean(unknown < t)) / 2, t)))


def reader_metrics(scores, rows, threshold):
    results = []
    for row, values in zip(rows, scores):
        best = int(np.argmax(values))
        label = row['label']
        margin = None if label is None else float(values[label] - np.max(np.delete(values, label)))
        results.append(dict(row, predicted=best, accepted=bool(values[best] >= threshold),
                            correct=label is not None and best == label, margin=margin,
                            scores=values.tolist()))
    known = [r for r in results if r['label'] is not None]
    unknown = [r for r in results if r['label'] is None]
    accuracy = float(np.mean([r['correct'] for r in known]))
    by_relation = {rel: float(np.mean([r['correct'] for r in known if r['relation'] == rel]))
                   for rel in ['capital', 'currency', 'language']}
    false_accept = float(np.mean([r['accepted'] for r in unknown])) if unknown else None
    summary = dict(n=len(known), accuracy=accuracy, by_relation=by_relation,
                   accepted_correct=float(np.mean([r['correct'] and r['accepted'] for r in known])),
                   unknown_n=len(unknown), false_acceptance=false_accept,
                   threshold=threshold,
                   gate=bool(accuracy >= .8 and np.mean([r['correct'] and r['accepted'] for r in known]) >= .8
                        and min(by_relation.values()) >= .75 and
                        false_accept is not None and false_accept <= .1))
    return summary, results


def fit_readers(dev_rows, dev_x):
    def subset(group):
        ids = [i for i, row in enumerate(dev_rows) if row['group'] == group]
        return [dev_rows[i] for i in ids], dev_x[ids]
    _, keys = subset('enroll')
    train_rows, train = subset('train')
    valid_rows, valid = subset('validation')
    _, unknown = subset('unknown')
    trials = []
    models = {}
    for lam in [.01, .1, 1., 10.]:
        model = RidgeMatch(train, keys, [r['label'] for r in train_rows], lam)
        accuracy = float(np.mean(model.scores(valid, keys).argmax(1) == [r['label'] for r in valid_rows]))
        trials.append(dict(regularization=lam, validation_accuracy=accuracy))
        models[lam] = model
    selected = max(trials, key=lambda r: (r['validation_accuracy'], r['regularization']))
    probe = models[selected['regularization']]
    cosine = lambda q, k: unit_rows(q) @ unit_rows(k).T
    readers = {'nearest_address': cosine, 'ridge_match': probe.scores}
    thresholds = {name: choose_threshold(fn(valid, keys).max(1), fn(unknown, keys).max(1))
                  for name, fn in readers.items()}
    return readers, thresholds, dict(trials=trials, selected=selected)


def evaluate_roster(readers, thresholds, rows, x):
    keys = x[[i for i, r in enumerate(rows) if r['group'] == 'enroll']]
    indices = [i for i, r in enumerate(rows) if r['group'] != 'enroll']
    return {name: reader_metrics(fn(x[indices], keys), [rows[i] for i in indices], thresholds[name])
            for name, fn in readers.items()}
