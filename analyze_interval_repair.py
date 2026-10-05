"""Compare pointwise and interval repair on the APRNN tutorial's 1-D ReLU toy network.

Run from the root of DNNProvableRepairTutorial at the pinned commit below. This
script uses the group's SyTorch/Gurobi repair code; the independent interval
check exploits the toy network's exact Linear(1,3)-ReLU-Linear(3,1) structure.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
import sytorch as st
from tutorial_helpers import get_dnn


UPSTREAM_COMMIT = "891b228b3eef4293c937d6d5b931b71871b6fe4f"
LOW, HIGH, REFERENCE = 1.0, 3.0, 1.5
TOLERANCE = 1e-7


def check_checkout():
    if not Path("tutorial_helpers.py").is_file():
        raise RuntimeError("Run from the root of DNNProvableRepairTutorial")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError("Expected tutorial commit " + UPSTREAM_COMMIT + ", got " + commit)
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Tutorial checkout is modified; use a clean copy")
    return commit


def parameters(network):
    if (len(network) != 3 or tuple(network[0].weight.shape) != (3, 1)
            or not isinstance(network[0], st.nn.Linear)
            or not isinstance(network[1], st.nn.ReLU)
            or not isinstance(network[2], st.nn.Linear)
            or tuple(network[0].bias.shape) != (3,)
            or tuple(network[2].weight.shape) != (1, 3)
            or tuple(network[2].bias.shape) != (1,)):
        raise RuntimeError("The exact knot check only applies to the tutorial's 1-3-1 network")
    with torch.no_grad():
        return {
            "first_weight": network[0].weight[:, 0].detach().cpu().tolist(),
            "first_bias": network[0].bias.detach().cpu().tolist(),
            "last_weight": network[2].weight[0, :].detach().cpu().tolist(),
            "last_bias": float(network[2].bias[0]),
        }


def scalar_output(network, x):
    with torch.no_grad(), st.no_symbolic():
        return float(network(torch.tensor([[x]], dtype=torch.float64))[0, 0])


def exact_interval_check(network, bound):
    """Enumerate all candidate extrema for this topology; evaluate in float64."""
    p = parameters(network)
    roots = []
    for weight, bias in zip(p["first_weight"], p["first_bias"]):
        if weight != 0.0:
            root = -bias / weight
            if LOW < root < HIGH:
                roots.append(root)
    knots = sorted(set([LOW, HIGH] + roots))
    values = [scalar_output(network, x) for x in knots]
    # Validate the expected piecewise-affine behavior between every pair of knots.
    for left, right, y_left, y_right in zip(knots, knots[1:], values, values[1:]):
        mid = (left + right) / 2
        observed = scalar_output(network, mid)
        if not math.isclose(observed, (y_left + y_right) / 2, abs_tol=TOLERANCE):
            raise AssertionError("Network is not affine between enumerated knots")
    return {
        "knots": knots,
        "outputs_at_knots": values,
        "minimum": min(values),
        "maximum": max(values),
        "all_interval_outputs_within_bound": all(abs(y) <= bound + TOLERANCE for y in values),
        "reason": "For Linear(1,3)-ReLU-Linear(3,1), output is affine between first-layer ReLU roots; extrema on [1,3] occur at those roots or interval endpoints.",
    }


def repair(bound, mode):
    network = get_dnn().to(dtype=torch.float64)
    original_shapes = [tuple(p.shape) for p in network.parameters()]
    solver = st.GurobiSolver().verbose_(False)
    solver.solver.Params.Threads = 1
    network.to(solver).repair()
    network.requires_symbolic_weight_and_bias()
    points = torch.tensor([[LOW], [HIGH]], dtype=torch.float64)
    if mode == "pointwise":
        symbolic = network(points)
    else:
        reference_points = torch.tensor([[REFERENCE], [REFERENCE]], dtype=torch.float64)
        shared_pattern = network.activation_pattern(reference_points)
        symbolic = network(points, pattern=shared_pattern)
    # Reuse this forward pass: delta(points) alone would encode an additional
    # forward with the original pointwise patterns, constraining interval repair.
    objective_expression = network.delta(points, sym_output=symbolic)
    feasible = solver.solve(-bound <= symbolic, symbolic <= bound, minimize=objective_expression)
    if not feasible:
        raise AssertionError("Repair unexpectedly infeasible: " + mode + " " + str(bound))
    objective = float(solver.solver.ObjVal)
    network.update_().repair(False).eval()
    if [tuple(p.shape) for p in network.parameters()] != original_shapes:
        raise AssertionError("Network architecture changed")
    interval = exact_interval_check(network, bound)
    endpoints = [scalar_output(network, LOW), scalar_output(network, HIGH)]
    if not all(abs(y) <= bound + TOLERANCE for y in endpoints):
        raise AssertionError("Solver result violates constrained endpoint")
    result = {
        "mode": mode,
        "bound": bound,
        "solver_feasible": feasible,
        "solver_objective": objective,
        "solver_status": solver.solver.Status,
        "solver_threads": solver.solver.Params.Threads,
        "solver_variables": solver.solver.NumVars,
        "solver_constraints": solver.solver.NumConstrs,
        "endpoint_outputs": endpoints,
        "parameters": parameters(network),
        "exact_interval_check": interval,
    }
    if mode == "shared_pattern" and not interval["all_interval_outputs_within_bound"]:
        raise AssertionError("Shared-pattern repair failed independent interval check")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    commit = check_checkout()
    baseline = get_dnn().to(dtype=torch.float64).eval()
    result = {
        "upstream_commit": commit,
        "environment": {"torch": torch.__version__, "gurobi": ".".join(map(str, __import__("gurobipy").gurobi.version()))},
        "interval": [LOW, HIGH],
        "reference_for_shared_pattern": REFERENCE,
        "objective": "output-delta norm plus parameter-delta norm; each norm is Linf plus normalized L1",
        "verification_scope": "Complete extrema enumeration for this 1-D, one-hidden-layer topology, evaluated in float64 with tolerance 1e-7; not an exact-arithmetic certificate or a verifier for general DNNs",
        "baseline": {**exact_interval_check(baseline, 0.1), "parameters": parameters(baseline)},
        "runs": [repair(bound, mode) for bound in (0.1, 0.02) for mode in ("pointwise", "shared_pattern")],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "baseline_max": result["baseline"]["maximum"],
        "runs": [{"bound": r["bound"], "mode": r["mode"],
                  "max": r["exact_interval_check"]["maximum"],
                  "interval_ok": r["exact_interval_check"]["all_interval_outputs_within_bound"]}
                 for r in result["runs"]],
    }, indent=2))


if __name__ == "__main__":
    main()
