"""Generate the large-scale scenario of the paper (20 tasks, IS/OD interleaved) as a scenario JSON."""
import argparse
import json

import numpy as np

CLASSES = ["dining_table", "dog", "horse", "motorbike", "person"]


def large_scale(model_file, rate, n_tasks=20, seed=0, training_budget=500000.0):
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_tasks):
        k = i // 2                      # index within its CV method
        is_task = i % 2 == 0            # odd task ids (1, 3, ...) are instance segmentation
        tasks.append({
            "id": f"task{i + 1}",
            "cv_method": "is" if is_task else "od",
            "object_class": CLASSES[k % 5],
            "priority": round((1.0 if is_task else 0.96) - 0.1 * k, 2),
            "min_accuracy": round((0.2 if is_task else 0.143) - 0.014 * k, 3),
            "max_latency_s": round((0.26 if is_task else 0.22) + 0.02 * k, 2),
            "request_rate_hz": rate,
            "path_loss_db": float(rng.choice([0, 5, 10, 15, 20, 25])),
        })
    return {
        "models": [model_file],
        "tasks": tasks,
        "resources": {"radio_rbs": 50, "compute_s_per_s": 5.0, "memory_mib": 12000.0,
                      "training_budget_s": training_budget},
        "radio": {
            "rb_bandwidth_hz": 200e3,
            "reference_snr_db": 30.0,
            "input_size_bits": [880e3, 176e3, 64.8e3],
            "latency_model": {"intercept_s": 0.0818, "per_rb_s": -0.00376, "per_request_rate_s": 0.0009,
                              "per_path_loss_db_s": 0.0074, "per_quality_index_s": -0.0055},
        },
        "alpha": 0.5,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_file")
    ap.add_argument("out")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(large_scale(a.model_file, a.rate, a.tasks, a.seed), f, indent=1)
    print("written", a.out)
