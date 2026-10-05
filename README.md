# APRNN toy network: pointwise versus interval repair

This experiment compares pointwise and interval repair using UC Davis DARG's public [DNNProvableRepairTutorial](https://github.com/95616ARG/DNNProvableRepairTutorial) at commit `891b228b3eef4293c937d6d5b931b71871b6fe4f`. Repair uses the tutorial's SyTorch/Gurobi implementation.

`analyze_interval_repair.py` compares repairs at the two endpoints of `[1,3]` with a shared-activation-pattern repair for that interval on the tutorial's one-input, three-ReLU network. It repeats the comparison for output bounds ±0.1 and ±0.02. After each solve it reads the repaired first-layer parameters, enumerates every ReLU root inside `[1,3]`, and checks the endpoints and roots. For this **specific one-hidden-layer 1-D network**, those are all possible extrema of the continuous interval, up to floating-point tolerance. `interval-analysis.json` records the outputs, parameters, solver objective, and knot checks. `verify_recorded_results.py` independently recomputes the knot outputs from that JSON using only Python's standard library.

The pinned `requires_symbolic_weight_and_bias()` helper edits the **first layer's weights and biases plus the output bias**; output weights remain fixed. With a fixed activation pattern, this avoids products of two solver variables, so the repair is a linear program. Its objective is the sum of separate penalties on endpoint-output changes and parameter changes. Each penalty is the maximum absolute change plus the mean absolute change. The shared-pattern run constrains both endpoints to the original pattern at input 1.5. This makes each hidden activation, and thus the repaired output, affine across the interval.

The objective explicitly reuses the constrained symbolic output via `delta(points, sym_output=symbolic)`. Calling `delta(points)` would perform an additional forward with the original endpoint patterns, adding unintended constraints to the shared-pattern comparison. The experiment uses one CPU thread for PyTorch and each solve.

To reproduce in a clean tutorial checkout with the tutorial's Python dependencies and a working Gurobi license:

```sh
git clone https://github.com/95616ARG/DNNProvableRepairTutorial.git
cd DNNProvableRepairTutorial
git checkout 891b228b3eef4293c937d6d5b931b71871b6fe4f
PYTHONPATH=. python /path/to/analyze_interval_repair.py --output /path/to/interval-analysis.json
python /path/to/verify_recorded_results.py /path/to/interval-analysis.json
```

On the corrected Mac arm64 run (Python 3.9.7, PyTorch 1.11.0, Gurobi 12.0.3), the original network reached 0.5 inside `[1,3]`. For the ±0.1 requirement, pointwise repair made both endpoints compliant but still reached ≈0.2467 at input ≈1.7778; shared-pattern repair stayed within ±0.1 at every enumerated knot. For the ±0.02 requirement, the corresponding maxima were ≈0.2459 and ≈0.02. All four runs reached Gurobi's optimal status, with 37 variables and 56 constraints each. The independent checker also reproduces the objective from the recorded baseline and repaired parameters. The repaired network remains the same 1-3-1 architecture.

`build_repair_comparison.py` and `repair-comparison.json` contain a separate two-bound, **pointwise-only** comparison at inputs `-1.5` and `-0.5`. The script executes the pinned `tutorial.py` with different constraint strings. Its assertions apply only to those two inputs.

The added comparison, recorded diagnostics, and topology-specific checker make a reproducible teaching experiment. The point-versus-region distinction is already in the papers and tutorial; this is not a new repair algorithm, a large-network evaluation, or an implementation of PREPARED. The complete candidate-extrema set follows analytically from this toy architecture, but its numerical evaluation uses float64 and a 1e-7 tolerance, not exact arithmetic. The relevant papers are [Architecture-Preserving Provable Repair of Deep Neural Networks (PLDI 2023)](https://thakur.cs.ucdavis.edu/assets/pubs/PLDI2023.pdf) and [Provable Editing of Deep Neural Networks using Parametric Linear Relaxation (NeurIPS 2024)](https://thakur.cs.ucdavis.edu/assets/pubs/NeurIPS2024.pdf).
