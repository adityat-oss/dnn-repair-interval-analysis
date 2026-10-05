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


def check(path):
    record = json.loads(Path(path).read_text())
    lower, upper = record["interval"]
    assert (lower, upper) == (1.0, 3.0)
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
        # Dense samples are a cross-check; extrema still come from the knots.
        for i in range(2001):
            y = output(p, lower + (upper - lower) * i / 2000)
            assert saved["minimum"] - 1e-9 <= y <= saved["maximum"] + 1e-9
    return len(record["runs"])


if __name__ == "__main__":
    count = check(sys.argv[1])
    print("verified", count, "runs")
