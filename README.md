# OffloaDNN — DOT solver

Reference implementation of the OffloaDNN heuristic for the *DNNs for scalable Offloading of
Tasks* (DOT) problem: given a set of computer-vision tasks offloaded to the edge, it selects the
DNN path and the input data quality of each task, the task admission ratios and the radio
resource blocks, accounting for the memory saved by sharing DNN blocks across tasks.

```bash
pip install numpy gurobipy            # a Gurobi license is required
python -m dot_solver examples/scenario_yolov5x.json -o result.json
```

```python
from dot_solver import solve
result = solve(scenario)              # scenario: dictionary in the format below
```

## How it works

1. **Branch selection** (`select_branch`). For each task, by decreasing priority, the candidate
   paths are those meeting its accuracy requirement. Paths already used by higher-priority tasks
   come first; the others are ranked by training cost amortized over the tasks that could use
   them. The first combination that fits the memory budget is selected, and each task gets the
   lowest input quality that still meets its accuracy requirement. This stage depends only on
   the tasks and on the memory budget: run it when a task is added or removed.
2. **Resource allocation** (`allocate`). With paths and qualities fixed, the admission ratios and
   the resource blocks minimizing the DOT cost are computed with Gurobi for the current channel
   conditions. Run it at each control loop iteration.

## Scenario file

```jsonc
{
  "models": ["yolov5x.json"],          // model catalogs: file names (relative to this file) or inline objects
  "tasks": [
    {"id": "task1",
     "cv_method": "is",                // must match a CV method of the model catalog
     "object_class": "dining_table",
     "priority": 1.0,                  // weight of the task in the admission term
     "min_accuracy": 0.2,
     "max_latency_s": 0.26,            // end-to-end: network + inference
     "request_rate_hz": 10.0,
     "path_loss_db": 15.0}             // current channel state of the device
  ],
  "resources": {"radio_rbs": 50, "compute_s_per_s": 5.0, "memory_mib": 12000.0,
                "training_budget_s": 500000.0},   // training cost is normalized to this budget
  "radio": {
    "rb_bandwidth_hz": 200000.0,
    "reference_snr_db": 30.0,          // SNR at 0 dB path loss
    "input_size_bits": [880000.0, 176000.0, 64800.0],   // per quality index, 0 = highest quality
    "latency_model": {"intercept_s": 0.0818, "per_rb_s": -0.00376, "per_request_rate_s": 0.0009,
                      "per_path_loss_db_s": 0.0074, "per_quality_index_s": -0.0055}
  },
  "alpha": 0.5                         // weight of task admission vs resource consumption
}
```

## Model catalog

```jsonc
{
  "name": "Yolov5x",
  "task_head_memory_mib": {"od": 351.0, "is": 401.0},   // per admitted task, never shared (0 if the head is part of a block)
  "blocks": [{"id": "b0", "memory_mib": 135.0}],        // a block used by several tasks is loaded once
  "paths": [
    {"id": "prune0/shared/all_frozen",
     "training_cost_s": 15450.0,
     "cv_methods": {
       "od": {"blocks": ["b0", "b1", "b2", "b3"],       // two paths share memory through common block ids
              "inference_time_s": 0.0214,
              "accuracy": {"dining_table": [0.492, 0.37, 0.172]}},   // per quality index, 0 = highest
       "is": {"blocks": ["b0", "b1", "b2", "b3"], "inference_time_s": 0.0341,
              "accuracy": {"dining_table": [0.219, 0.208, 0.104]}}
     }}
  ]
}
```

A path can list different blocks for different CV methods, e.g., when the last block includes a
prediction head that is specific to detection or to segmentation (`b<N>/od`, `b<N>/is`).
Several models can be listed in the same scenario; each task then uses the best path across all of them.

## Result

```jsonc
{
  "feasible": true,
  "decisions": [{"task": "task1", "model": "Yolov5x", "path": "prune85/dining_table/frozen_0",
                 "quality_level": 2,                    // number of levels = highest quality, 1 = lowest
                 "admission_ratio": 0.72, "resource_blocks": 9}],
  "metrics": {"dot_cost": 2.07, "weighted_admission_ratio": 9.47, "radio_usage": 0.53,
              "compute_usage": 0.65, "memory_mib": 7802.0, "training_cost_s": 10919.0},
  "timing_ms": {"branch_selection": 1.3, "resource_allocation": 58.2}
}
```

## Contents

| Path | Purpose |
|---|---|
| `dot_solver/solver.py` | the solver (data model, branch selection, resource allocation) |
| `dot_solver/__main__.py` | command line front end |
| `examples/yolov5x.json` | model catalog of YOLOv5x used in the paper |
| `examples/scenario_yolov5x.json` | large-scale scenario of the paper (20 tasks, medium arrival rate) |
| `examples/result_yolov5x.json` | output of the solver for `examples/scenario_yolov5x.json` |
| `tools/make_scenario.py` | generates the large-scale scenario for a given arrival rate and channel seed |
| `LICENSE.md` | license terms |

## License

This software is released under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): it can be
used, modified and redistributed for noncommercial purposes, including research and teaching.
Any commercial use requires a prior agreement with the authors: please contact
Corrado Puligheddu (corrado.puligheddu@polito.it).

Required Notice: Copyright (c) 2026 Corrado Puligheddu, Nancy Varshney, Tanzil Hassan, Jonathan Ashdown, Francesco Restuccia, Carla Fabiana Chiasserini

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{puligheddu2024offloadnn,
  author    = {Puligheddu, Corrado and Varshney, Nancy and Hassan, Tanzil and Ashdown, Jonathan and Restuccia, Francesco and Chiasserini, Carla Fabiana},
  title     = {{OffloaDNN}: Shaping {DNNs} for Scalable Offloading of Computer Vision Tasks at the Edge},
  booktitle = {2024 IEEE 44th International Conference on Distributed Computing Systems (ICDCS)},
  year      = {2024},
  pages     = {624--634},
  doi       = {10.1109/ICDCS60910.2024.00064}
}
```
