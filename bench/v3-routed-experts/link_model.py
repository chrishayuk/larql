"""DEC-V3-1 link model: pure analysis, no processes and no model.

The composed hypothesis under test, per decode position, for a shaped arm
measured against its unshaped relay control ("floor"):

    delta = a * crossings * rtt  +  b * critical_bits / rate  +  c

`crossings` is the number of sequential fan-outs a position waits for,
`critical_bits` the request plus response bits of the largest shard exchange
in each of those fan-outs. Shards are contacted in parallel on separate
links, so within a layer only the slowest exchange is on the critical path.
An exact additive link gives a = b = 1 and c = 0; the fit reports what the
measurements say instead.
"""
import statistics

NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000
BITS_PER_BYTE = 8
UNLIMITED = "inf"

# Same tolerance as PROFILE-1's control gate: two floors bracketing a point
# must agree this closely, and the relay's own schedule error must stay
# below it, or the point cannot resolve an effect of that size.
GATE = 0.01

FLOOR = {"kind": "floor", "rtt_ms": 0.0, "rate_bits_per_s": UNLIMITED}
DIRECT = {"kind": "direct", "rtt_ms": 0.0, "rate_bits_per_s": UNLIMITED}

# Frozen sweep (LINK-1.md). Axis points identify one coefficient each and are
# the only points the model is fitted on; joint points are held out and test
# whether the two terms compose.
RTT_AXIS_MS = [0.2, 1.0, 5.0, 20.0]
RATE_AXIS_BITS_PER_S = [1e9, 1e8]
JOINT_POINTS = [(1.0, 1e8), (5.0, 1e9), (20.0, 1e8)]
POINTS_PER_BRACKET = 2


def frozen_points():
    points = [dict(DIRECT)]
    points += [{"kind": "axis", "rtt_ms": rtt, "rate_bits_per_s": UNLIMITED} for rtt in RTT_AXIS_MS]
    points += [{"kind": "axis", "rtt_ms": 0.0, "rate_bits_per_s": rate} for rate in RATE_AXIS_BITS_PER_S]
    points += [{"kind": "joint", "rtt_ms": rtt, "rate_bits_per_s": rate} for rtt, rate in JOINT_POINTS]
    return points


def schedule(points, repeats, per_bracket=POINTS_PER_BRACKET):
    """Floor-bracketed order; odd repeats run reversed so drift is not aliased to RTT."""
    if repeats < 1 or per_bracket < 1 or not points:
        raise ValueError("need at least one point, one repeat and one point per bracket")
    order = []
    for repeat in range(repeats):
        sequence = list(points) if repeat % 2 == 0 else list(reversed(points))
        order.append(dict(FLOOR, repeat=repeat))
        for start in range(0, len(sequence), per_bracket):
            order += [dict(point, repeat=repeat) for point in sequence[start:start + per_bracket]]
            order.append(dict(FLOOR, repeat=repeat))
    return order


def one_way_delay_us(point):
    """The relay delays each direction by half the configured round trip."""
    return round(point["rtt_ms"] * 1000 / 2)


def link_shape(rows, layers):
    """Sequential crossings and critical-path bits per measured position."""
    if not rows:
        raise ValueError("no measured decode positions")
    crossings, critical_bits, total_bytes = [], [], []
    for row in rows:
        calls = row["provider_calls"]
        position_bits, position_bytes, position_crossings = 0, 0, 0
        for layer in range(layers):
            exchanges = [
                call["transport"][0]["request_bytes"] + call["transport"][0]["response_bytes"]
                for call in calls
                if call.get("kind") == "expert_shard" and call["layer"] == layer
            ]
            if not exchanges:
                raise ValueError(f"layer {layer} made no remote exchange")
            position_crossings += 1
            position_bits += max(exchanges) * BITS_PER_BYTE
            position_bytes += sum(exchanges)
        crossings.append(position_crossings)
        critical_bits.append(position_bits)
        total_bytes.append(position_bytes)
    return {
        "crossings": statistics.mean(crossings),
        "critical_bits": statistics.mean(critical_bits),
        "body_bytes": statistics.mean(total_bytes),
    }


def regressors(point, shape):
    """The two model terms in nanoseconds, before any coefficient."""
    rtt_ns = shape["crossings"] * point["rtt_ms"] * NS_PER_MS
    rate = point["rate_bits_per_s"]
    rate_ns = 0.0 if rate == UNLIMITED else shape["critical_bits"] / rate * NS_PER_SECOND
    return rtt_ns, rate_ns


def relay_gate(stats, positions, floor_total_ns):
    """Reject a trial whose relay released chunks later than the timing gate resolves."""
    if not stats:
        raise ValueError("relay recorded no traffic")
    late_sum = sum(record["late_sum_ns"] for record in stats)
    chunks = sum(record["chunks"] for record in stats)
    if chunks == 0:
        raise ValueError("relay forwarded no chunks")
    late_per_position = late_sum / positions
    return {
        "relay_chunks": chunks,
        "relay_bytes": sum(record["bytes"] for record in stats),
        "relay_late_mean_ns": late_sum / chunks,
        "relay_late_max_ns": max(record["late_max_ns"] for record in stats),
        # Upper bound: lateness on parallel shards overlaps, so the critical
        # path sees at most this much unscheduled delay per position.
        "relay_late_per_position_ns": late_per_position,
        "relay_on_schedule": late_per_position <= GATE * floor_total_ns,
    }


def solve(matrix, vector):
    """Gaussian elimination with partial pivoting; refuses a singular system."""
    size = len(vector)
    rows = [list(matrix[i]) + [vector[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) == 0:
            raise ValueError("observations do not identify every coefficient")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(column + 1, size):
            factor = rows[row][column] / rows[column][column]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
    solution = [0.0] * size
    for row in reversed(range(size)):
        known = sum(rows[row][k] * solution[k] for k in range(row + 1, size))
        solution[row] = (rows[row][size] - known) / rows[row][row]
    return solution


def fit(observations):
    """Least squares for delta = a * rtt_ns + b * rate_ns + c over axis points."""
    design = [[obs["rtt_ns"], obs["rate_ns"], 1.0] for obs in observations]
    target = [obs["delta_ns"] for obs in observations]
    width = len(design[0]) if design else 0
    if len(design) < width:
        raise ValueError("fewer observations than coefficients")
    normal = [[sum(row[i] * row[j] for row in design) for j in range(width)] for i in range(width)]
    moment = [sum(row[i] * y for row, y in zip(design, target)) for i in range(width)]
    a, b, c = solve(normal, moment)
    residuals = [y - (a * row[0] + b * row[1] + c) for row, y in zip(design, target)]
    mean = statistics.mean(target)
    spread = sum((y - mean) ** 2 for y in target)
    return {
        "rtt_coefficient": a,
        "rate_coefficient": b,
        "intercept_ns": c,
        "observations": len(target),
        "residual_ns": residuals,
        "r_squared": 1 - sum(r * r for r in residuals) / spread if spread > 0 else None,
    }


def predict(model, observation):
    return (model["rtt_coefficient"] * observation["rtt_ns"]
            + model["rate_coefficient"] * observation["rate_ns"] + model["intercept_ns"])


def analyse(trials):
    """Bracket every non-floor trial by its neighbouring floors, then fit and hold out.

    `trials` is the executed schedule in order. A point is valid only when its
    two floors agree within the gate, its relay stayed on schedule and it kept
    token parity (the runner refuses otherwise, so parity is not re-derived).
    """
    floors = [i for i, trial in enumerate(trials) if trial["point"]["kind"] == "floor"]
    observations = []
    for index, trial in enumerate(trials):
        point = trial["point"]
        if point["kind"] == "floor":
            continue
        before = max((i for i in floors if i < index), default=None)
        after = min((i for i in floors if i > index), default=None)
        if before is None or after is None:
            raise ValueError("every point needs a floor before and after it")
        low, high = trials[before]["total_ns"], trials[after]["total_ns"]
        drift = abs(low - high) / ((low + high) / 2)
        floor = (low + high) / 2
        rtt_ns, rate_ns = regressors(point, trial["shape"])
        on_schedule = trial.get("relay", {}).get("relay_on_schedule", point["kind"] == "direct")
        observations.append({
            "index": index, "point": point, "repeat": point["repeat"],
            "total_ns": trial["total_ns"], "floor_ns": floor, "floor_drift": drift,
            "delta_ns": trial["total_ns"] - floor,
            "ffn_delta_ns": trial["ffn_ns"] - (trials[before]["ffn_ns"] + trials[after]["ffn_ns"]) / 2,
            "rtt_ns": rtt_ns, "rate_ns": rate_ns,
            "positions_per_second": NS_PER_SECOND / trial["total_ns"],
            "valid": drift <= GATE and on_schedule,
        })
    axis = [o for o in observations if o["valid"] and o["point"]["kind"] == "axis"]
    joint = [o for o in observations if o["valid"] and o["point"]["kind"] == "joint"]
    report = {"observations": observations, "valid": sum(o["valid"] for o in observations),
              "void": sum(not o["valid"] for o in observations), "model": None, "held_out": []}
    try:
        model = fit(axis)
    except ValueError as error:
        report["model_refused"] = str(error)
        return report
    report["model"] = model
    for obs in joint:
        predicted = predict(model, obs)
        report["held_out"].append({
            "point": obs["point"], "measured_delta_ns": obs["delta_ns"],
            "predicted_delta_ns": predicted,
            "error_fraction_of_position": (obs["delta_ns"] - predicted) / obs["total_ns"],
        })
    return report
