"""Command line front end:  python -m dot_solver scenario.json [-o result.json]"""
import argparse
import json
import os
import sys

from .solver import solve


def load_scenario(path):
    """Load a scenario; models given as file names are read relative to the scenario file."""
    with open(path) as f:
        scenario = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    models = []
    for m in scenario["models"]:
        if isinstance(m, str):
            with open(os.path.join(base, m)) as f:
                m = json.load(f)
        models.append(m)
    scenario["models"] = models
    return scenario


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dot_solver", description="Solve a DOT scenario with OffloaDNN.")
    ap.add_argument("scenario", help="scenario JSON file")
    ap.add_argument("-o", "--output", help="result JSON file (default: standard output)")
    ap.add_argument("--threads", type=int, default=0, help="Gurobi threads (0 = automatic)")
    a = ap.parse_args(argv)

    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", a.threads)
    env.start()

    result = solve(load_scenario(a.scenario), env)
    text = json.dumps(result, indent=2)
    if a.output:
        with open(a.output, "w") as f:
            f.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0 if result["feasible"] else 1


if __name__ == "__main__":
    sys.exit(main())
