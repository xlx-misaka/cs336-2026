import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import curve_fit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/isoflops_curves.json"))
    parser.add_argument("--output", type=Path, default=Path("report/results"))
    args = parser.parse_args()
    return args


def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def selct_optim(rows):
    loss_params_per_cpm = {}
    for row in rows:
        loss_params_per_cpm[row["compute_budget"]] = min(
            loss_params_per_cpm.get(
                row["compute_budget"], (float("inf"), float("inf"))
            ),
            (row["final_loss"], row["parameters"]),
        )
    results = [
        {
            "parameters": parameters,
            "tokens": compute_budget / (6 * parameters),
            "compute_budget": compute_budget,
            "final_loss": final_loss,
        }
        for compute_budget, (final_loss, parameters) in loss_params_per_cpm.items()
    ]
    return results


def power(x, coefficient, exponent):
    return coefficient * x**exponent


def save_figure(figure, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_power(x, y, coefficient, exponent, targets, y_label, output_path):
    grid = np.logspace(np.log10(x.min()), 24, 300)
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    axis.scatter(
        x,
        y,
        s=55,
        marker="o",
        color="tab:blue",
        label="Discrete minima",
    )
    axis.plot(
        grid,
        coefficient * grid**exponent,
        label=f"fit exponent={exponent:.3f}",
    )
    target_predictions = coefficient * targets**exponent
    axis.scatter(
        targets,
        target_predictions,
        s=80,
        marker="s",
        color="tab:orange",
        label="Extrapolated predictions",
    )
    for x_value, y_value in zip(targets, target_predictions):
        axis.annotate(
            f"{y_value:.3e}",
            xy=(x_value, y_value),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(xlabel="Compute C (FLOPs)", ylabel=y_label)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    save_figure(figure, output_path)


def run():
    args = parse_args()

    rows = load_data(args.data)

    rows = selct_optim(rows)

    compute_budget = np.asarray([row["compute_budget"] for row in rows])
    tokens = np.asarray([row["tokens"] for row in rows])
    parameters = np.asarray([row["parameters"] for row in rows])
    targets = np.asarray([1e23, 1e24])

    (n_coefficient, n_exponent), _ = curve_fit(
        power,
        compute_budget,
        parameters,
    )

    (t_coefficient, t_exponent), _ = curve_fit(
        power,
        compute_budget,
        tokens,
    )

    plot_power(
        compute_budget,
        parameters,
        n_coefficient,
        n_exponent,
        targets,
        "Compute-optimal parameters",
        args.output / "figures" / "n_opt_compute",
    )

    plot_power(
        compute_budget,
        tokens,
        t_coefficient,
        t_exponent,
        targets,
        "Compute-optimal tokens",
        args.output / "figures" / "t_opt_compute",
    )


if __name__ == "__main__":
    run()
