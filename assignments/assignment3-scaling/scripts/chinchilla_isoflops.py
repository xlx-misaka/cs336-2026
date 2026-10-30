import argparse
import json
import numpy as np

from pathlib import Path
from scipy.optimize import curve_fit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/isoflops_curves.json"))
    parser.add_argument("--results", type=Path, default=Path("report/results"))
    args = parser.parse_args()
    return args


def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def selct_optim(rows):
    loss_params_per_cpm = {}
    for row in rows:
        loss_params_per_cpm[row["parameters"]] = min(
            loss_params_per_cpm.get(row["parameters"], float("inf")),
            (rows["final_loss"], rows["compute_budget"]),
        )
    results = [
        {
            "parameters": parameters,
            "compute_budget": compute_budget,
            "final_loss": final_loss,
        }
        for parameters, (final_loss, compute_budget) in loss_params_per_cpm
    ]
    return results


def line(x, slope, intercept):
    return slope * x + intercept


def plot(rows):
    compute_budget = np.asarray([row["compute_budget"] for row in rows])
    parameters = np.asarray([row["parameters"] for row in rows])


def run():
    args = parse_args()

    rows = load_data(args.data)

    rows = selct_optim(rows)

    compute_budget = np.asarray([row["compute_budget"] for row in rows])
    parameters = np.asarray([row["parameters"] for row in rows])
    targets_x = np.asarray([1e23, 1e24])

    popt, _ = curve_fit(
        line,
        compute_budget,
        parameters,
    )

    slope, intercept = popt

    targets_y = line(targets_x, slope, intercept)

    rows += [
        {"compute_budget": compute_budget, "parameters": parameters, "final_loss": None}
        for compute_budget, parameters in zip(targets_y, targets_x)
    ]


if __name__ == "__main__":
    run()
