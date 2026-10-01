"""OffloaDNN: heuristic solver of the DOT (DNNs for scalable Offloading of Tasks) problem.

The solver works in two stages (Sec. IV of the paper):

1. Branch selection  -- for every task, pick a DNN path (a sequence of blocks of one model) and
   the input data quality. Depends only on the task set, the task requirements and the memory
   budget, so it is repeated only when a task is added or removed.
2. Resource allocation -- given the selected paths and qualities, compute the task admission
   ratios z and the resource blocks r for the current channel conditions (a small mixed-integer
   program solved with Gurobi). Repeated at each control loop iteration.

Inputs and outputs are plain dictionaries following the JSON formats described in the README.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

LATENCY_BIG_M = 10.0  # [s] larger than any latency the model can produce


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Path:
    """A DNN path usable by a given CV method: blocks, inference time and accuracy."""
    model: str
    path_id: str
    training_cost_s: float
    blocks: tuple            # block ids (unique within the model)
    inference_time_s: float
    accuracy: tuple          # accuracy for the task's object class, one value per quality level
    key: tuple = field(compare=False, default=())  # (model, path_id): identifies the path across CV methods


@dataclass
class Task:
    id: str
    cv_method: str
    object_class: str
    priority: float
    min_accuracy: float
    max_latency_s: float
    request_rate_hz: float
    path_loss_db: float = 0.0


class Catalog:
    """Blocks and paths of the available DNN models."""

    def __init__(self, models: list[dict]):
        self.block_memory = {}     # (model, block id) -> MiB
        self.head_memory = {}      # (model, cv method) -> MiB per admitted task, not shareable
        self.models = models
        for m in models:
            for b in m["blocks"]:
                self.block_memory[(m["name"], b["id"])] = float(b["memory_mib"])
            for cv, mem in m.get("task_head_memory_mib", {}).items():
                self.head_memory[(m["name"], cv)] = float(mem)

    def paths_for(self, task: Task) -> list[Path]:
        """Paths able to serve the task's CV method and object class, in catalog order."""
        out = []
        for m in self.models:
            for p in m["paths"]:
                spec = p["cv_methods"].get(task.cv_method)
                if spec is None or task.object_class not in spec["accuracy"]:
                    continue
                out.append(Path(m["name"], p["id"], float(p["training_cost_s"]), tuple(spec["blocks"]),
                                float(spec["inference_time_s"]), tuple(spec["accuracy"][task.object_class]),
                                key=(m["name"], p["id"])))
        return out


# --------------------------------------------------------------------------------------
# Stage 1: branch selection
# --------------------------------------------------------------------------------------
def feasible_paths(catalog: Catalog, task: Task) -> list[Path]:
    """Vertices of the task's clique: accuracy met at the best quality, inference time within the deadline."""
    return [p for p in catalog.paths_for(task)
            if p.accuracy[0] >= task.min_accuracy and p.inference_time_s <= task.max_latency_s]


def order_cliques(cliques: list[list[Path]]) -> list[list[Path]]:
    """Vertex ordering (Algorithm 1): paths already used by higher-priority tasks first, by training
    cost; then the others, by training cost amortized over the tasks that could use them."""
    users = {}
    for clique in cliques:
        for p in clique:
            users[p.key] = users.get(p.key, 0) + 1
    ordered, used = [], set()
    for clique in cliques:
        common = sorted((p for p in clique if p.key in used), key=lambda p: p.training_cost_s)
        distinct = sorted((p for p in clique if p.key not in used),
                          key=lambda p: p.training_cost_s / users[p.key])
        clique = common + distinct
        ordered.append(clique)
        if clique:
            used.add(clique[0].key)
    return ordered


def memory_mib(catalog: Catalog, tasks: list[Task], branch: list[Path | None]) -> float:
    """Memory of a (partial) branch: every block counted once, plus one head per task."""
    seen, total = set(), 0.0
    for task, p in zip(tasks, branch):
        if p is None:
            continue
        for b in p.blocks:
            if (p.model, b) not in seen:
                seen.add((p.model, b))
                total += catalog.block_memory[(p.model, b)]
        total += catalog.head_memory.get((p.model, task.cv_method), 0.0)
    return total


def lowest_quality(task: Task, path: Path) -> int:
    """Index of the lowest quality level (largest index) whose accuracy exceeds the requirement."""
    ok = [q for q, a in enumerate(path.accuracy) if a > task.min_accuracy]
    return max(ok)


def select_branch(catalog: Catalog, tasks: list[Task], memory_budget_mib: float):
    """First memory-feasible branch of the tree (depth-first). Returns (paths, qualities) or None."""
    cliques = order_cliques([feasible_paths(catalog, t) for t in tasks])
    branch: list[Path | None] = [None] * len(tasks)

    def descend(i):
        if i == len(tasks):
            return True
        for p in cliques[i]:
            branch[i] = p
            if memory_mib(catalog, tasks, branch) <= memory_budget_mib and descend(i + 1):
                return True
            branch[i] = None
        return False

    if not descend(0):
        return None
    return branch, [lowest_quality(t, p) for t, p in zip(tasks, branch)]


# --------------------------------------------------------------------------------------
# Stage 2: resource allocation
# --------------------------------------------------------------------------------------
def allocate(tasks, branch, qualities, resources, radio, alpha, env=None):
    """Optimal admission ratios z and resource blocks r for the given branch and channel state.
    Returns (z, r, cost) where cost is the admission + radio + compute part of the DOT objective."""
    import gurobipy as gp

    T = len(tasks)
    R, C = resources["radio_rbs"], resources["compute_s_per_s"]
    lm = radio["latency_model"]
    size = [radio["input_size_bits"][q] for q in qualities]
    snr_db = [radio["reference_snr_db"] - t.path_loss_db for t in tasks]
    bitrate = [radio["rb_bandwidth_hz"] * np.log2(1 + 10 ** (s / 10)) for s in snr_db]
    lam = [t.request_rate_hz for t in tasks]
    comp = [p.inference_time_s for p in branch]

    m = gp.Model(env=env) if env is not None else gp.Model()
    z = m.addVars(T, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS)
    r = m.addVars(T, lb=0, ub=R, vtype=gp.GRB.INTEGER)
    u = m.addVars(T, vtype=gp.GRB.BINARY)          # u = 1 if the task is admitted (z > 0)
    m.setObjective(
        alpha * sum((1 - z[t]) * tasks[t].priority for t in range(T))
        + 0.5 * (1 - alpha) * sum(z[t] * lam[t] * (r[t] / R + comp[t] / C) for t in range(T)),
        gp.GRB.MINIMIZE)
    m.addConstr(sum(z[t] * r[t] for t in range(T)) <= R)                 # radio capacity
    m.addConstr(sum(z[t] * lam[t] * comp[t] for t in range(T)) <= C)      # compute capacity
    for t in range(T):
        m.addConstr(z[t] * lam[t] * size[t] <= bitrate[t] * r[t])        # enough bandwidth for the admitted rate
        m.addConstr(r[t] <= R * z[t])                                    # no RBs to rejected tasks
        m.addConstr(r[t] >= z[t])                                        # at least one RB to admitted tasks
        m.addConstr(z[t] <= u[t])
        latency = (lm["intercept_s"] + lm["per_rb_s"] * r[t] + lm["per_request_rate_s"] * z[t] * lam[t]
                   + lm["per_path_loss_db_s"] * tasks[t].path_loss_db
                   + lm["per_quality_index_s"] * qualities[t] + comp[t])
        m.addConstr(latency <= tasks[t].max_latency_s + LATENCY_BIG_M * (1 - u[t]))  # only if admitted
    m.setParam("IntFeasTol", 1e-9)
    m.setParam("NonConvex", 2)
    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"resource allocation not solved to optimality (Gurobi status {m.status})")
    zs, rs, cost = [z[t].X for t in range(T)], [r[t].X for t in range(T)], m.ObjVal
    m.dispose()
    return zs, rs, cost


# --------------------------------------------------------------------------------------
# Front end
# --------------------------------------------------------------------------------------
def parse_scenario(scenario: dict):
    catalog = Catalog(scenario["models"])
    tasks = [Task(**t) for t in scenario["tasks"]]
    tasks.sort(key=lambda t: -t.priority)      # the tree is built by decreasing priority
    return catalog, tasks


def solve(scenario: dict, env=None) -> dict:
    """Solve a scenario (see README for the format) and return the result dictionary."""
    catalog, tasks = parse_scenario(scenario)
    res, radio, alpha = scenario["resources"], scenario["radio"], scenario["alpha"]

    t0 = time.perf_counter()
    sel = select_branch(catalog, tasks, res["memory_mib"])
    t_sel = time.perf_counter() - t0
    if sel is None:
        return {"feasible": False, "reason": "no branch fits the memory budget"}
    branch, qualities = sel

    t0 = time.perf_counter()
    z, r, alloc_cost = allocate(tasks, branch, qualities, res, radio, alpha, env)
    t_alloc = time.perf_counter() - t0

    admitted = [zt > 1e-6 for zt in z]
    training = sum({p.key: p.training_cost_s for p in branch}.values())
    n_levels = len(radio["input_size_bits"])
    return {
        "feasible": True,
        "decisions": [
            {"task": t.id, "model": p.model, "path": p.path_id,
             "quality_level": n_levels - q,          # n_levels = highest quality, 1 = lowest
             "admission_ratio": zt, "resource_blocks": int(round(rt))}
            for t, p, q, zt, rt in zip(tasks, branch, qualities, z, r)],
        "metrics": {
            "dot_cost": alloc_cost + training / res["training_budget_s"],
            "weighted_admission_ratio": sum(zt * t.priority for zt, t in zip(z, tasks)),
            "radio_usage": sum(zt * rt for zt, rt in zip(z, r)) / res["radio_rbs"],
            "compute_usage": sum(zt * t.request_rate_hz * p.inference_time_s
                                 for zt, t, p in zip(z, tasks, branch)) / res["compute_s_per_s"],
            "memory_mib": memory_mib(catalog, tasks, [p if a else None for p, a in zip(branch, admitted)]),
            "training_cost_s": training,
        },
        "timing_ms": {"branch_selection": 1e3 * t_sel, "resource_allocation": 1e3 * t_alloc},
    }
