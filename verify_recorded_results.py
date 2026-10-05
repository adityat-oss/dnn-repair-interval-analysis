"""Independent, standard-library check of the recorded 1-3-1 ReLU toy results."""
import json
import math
import sys
from pathlib import Path


def output(parameters, x):
    return parameters["last_bias"] + sum(
        v * max(0.0, w * x + b)
        for w, b, v in zip(
            parameters["first_weight"],
            parameters["first_bias"],
            parameters["last_weight"],
        )
    )


def delta_norm(values):
    absolute = [abs(v) for v in values]
    return max(absolute) + sum(absolute) / len(absolute)


def check(path):
    record = json.loads(Path(path).read_text())
    lower, upper = record["interval"]
    assert (lower, upper) == (1.0, 3.0)
    assert {(r["mode"], r["bound"]) for r in record["runs"]} == {
        (mode, bound) for mode in ("pointwise", "shared_pattern") for bound in (0.1, 0.02)
    }
    assert len(record["runs"]) == 4
    baseline = record["baseline"].get("parameters")
    for run in record["runs"]:
        p = run["parameters"]
        knots = [lower, upper]
        for w, b in zip(p["first_weight"], p["first_bias"]):
            if w != 0.0 and lower < -b / w < upper:
                knots.append(-b / w)
        knots = sorted(set(knots))
        values = [output(p, x) for x in knots]
        saved = run["exact_interval_check"]
        assert len(knots) == len(saved["knots"])
        for actual, expected in zip(knots, saved["knots"]):
            assert math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(values, saved["outputs_at_knots"]):
            assert math.isclose(actual, expected, abs_tol=1e-9)
        assert math.isclose(max(values), saved["maximum"], abs_tol=1e-9)
        assert math.isclose(min(values), saved["minimum"], abs_tol=1e-9)
        assert all(abs(y) <= run["bound"] + 1e-7 for y in values) == saved["all_interval_outputs_within_bound"]
        endpoints = [output(p, lower), output(p, upper)]
        for actual, expected in zip(endpoints, run["endpoint_outputs"]):
            assert math.isclose(actual, expected, abs_tol=1e-9)
        assert all(abs(y) <= run["bound"] + 1e-7 for y in endpoints)
        if run["mode"] == "shared_pattern":
            assert saved["all_interval_outputs_within_bound"]
        if "solver_status" in run:
            assert run["solver_status"] == 2  # Gurobi OPTIMAL
        if baseline is not None:
            assert p["last_weight"] == baseline["last_weight"]
            changes = [a - b for key in ("first_weight", "first_bias")
                       for a, b in zip(p[key], baseline[key])]
            changes.append(p["last_bias"] - baseline["last_bias"])
            output_changes = [a - output(baseline, x)
                              for a, x in zip(endpoints, (lower, upper))]
            objective = delta_norm(changes) + delta_norm(output_changes)
            assert math.isclose(objective, run["solver_objective"], abs_tol=1e-6)
        # Dense samples are a cross-check; extrema still come from the knots.
        for i in range(2001):
            y = output(p, lower + (upper - lower) * i / 2000)
            assert saved["minimum"] - 1e-9 <= y <= saved["maximum"] + 1e-9
    return len(record["runs"])


if __name__ == "__main__":
    count = check(sys.argv[1])
    print("verified", count, "runs")
