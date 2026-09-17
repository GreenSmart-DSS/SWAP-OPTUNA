from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from metrics import RegressionMetrics
from vap import RequestedLayer


def plot_time_series(
    simulation: pd.DataFrame,
    observations: pd.DataFrame,
    metrics,
    output_path: Path,
    title_prefix: str = "Baseline",
) -> None:
    """Plot continuous SWAP simulation and field observations."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.8))

    ax.plot(
        simulation["date"],
        simulation["value"],
        linewidth=2.0,
        label="SWAP simulation",
    )

    ax.scatter(
        observations["date"],
        observations["value"],
        s=60,
        zorder=3,
        label="Field observations",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(r"Volumetric water content (cm$^3$ cm$^{-3}$)")
    ax.set_title(
        f"{title_prefix} SWAP Simulation and Field Observations " "(0–5 cm Soil Layer)"
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )

    ax.legend(
        frameon=True,
        loc="best",
    )

    metrics_text = (
        f"RMSE = {metrics.rmse:.6f}\n"
        f"MAE = {metrics.mae:.6f}\n"
        f"MBE = {metrics.mbe:.6f}\n"
        f"$R^2$ (Pearson$^2$) = {metrics.r2:.6f}\n"
        f"NSE = {metrics.nse:.6f}\n"
        f"n = {metrics.n}"
    )

    ax.text(
        0.015,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.9,
        },
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_scatter(
    matched: pd.DataFrame,
    metrics,
    output_path: Path,
    title_prefix: str = "Baseline",
) -> None:
    """Plot observed versus simulated soil water content."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 6.4))

    observed = matched["observed"]
    simulated = matched["simulated"]

    data_min = min(
        observed.min(),
        simulated.min(),
    )

    data_max = max(
        observed.max(),
        simulated.max(),
    )

    margin = 0.01

    axis_min = data_min - margin
    axis_max = data_max + margin

    ax.scatter(
        observed,
        simulated,
        s=65,
        zorder=3,
        label="Matched observations",
    )

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        linewidth=1.5,
        label="1:1 line",
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.set_xlabel(r"Observed volumetric water content (cm$^3$ cm$^{-3}$)")
    ax.set_ylabel(r"Simulated volumetric water content (cm$^3$ cm$^{-3}$)")

    ax.set_title(f"{title_prefix} SWAP Performance Against Field Observations")

    metrics_text = (
        f"$R^2$ (Pearson$^2$) = {metrics.r2:.6f}\n"
        f"NSE = {metrics.nse:.6f}\n"
        f"RMSE = {metrics.rmse:.6f}\n"
        f"MAE = {metrics.mae:.6f}\n"
        f"MBE = {metrics.mbe:.6f}\n"
        f"n = {metrics.n}"
    )

    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.9,
        },
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )

    ax.legend(
        frameon=True,
        loc="lower right",
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_layer_time_series(
    simulation: pd.DataFrame,
    observations: pd.DataFrame | None,
    metrics: RegressionMetrics | None,
    layer: RequestedLayer,
    output_path: Path,
    title_prefix: str = "SWAP",
) -> None:
    """Plot SWAP simulation for one soil layer with optional observations."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.8))

    ax.plot(
        simulation["date"],
        simulation["value"],
        linewidth=2.0,
        label="SWAP simulation",
    )

    if observations is not None:
        ax.scatter(
            observations["date"],
            observations["value"],
            s=60,
            zorder=3,
            label="Field observations",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel(r"Volumetric water content (cm$^3$ cm$^{-3}$)")

    if observations is not None:
        title = (
            f"{title_prefix} SWAP Simulation and Field Observations "
            f"({layer.name} cm Soil Layer)"
        )
    else:
        title = f"{title_prefix} SWAP Simulation " f"({layer.name} cm Soil Layer)"

    ax.set_title(title)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )

    ax.legend(
        frameon=True,
        loc="best",
    )

    if metrics is not None:
        metrics_text = (
            f"RMSE = {metrics.rmse:.6f}\n"
            f"MAE = {metrics.mae:.6f}\n"
            f"MBE = {metrics.mbe:.6f}\n"
            f"$R^2$ (Pearson$^2$) = {metrics.r2:.6f}\n"
            f"NSE = {metrics.nse:.6f}\n"
            f"n = {metrics.n}"
        )

        ax.text(
            0.015,
            0.97,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.9,
            },
        )

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_layer_scatter(
    matched: pd.DataFrame,
    metrics: RegressionMetrics,
    layer: RequestedLayer,
    output_path: Path,
    title_prefix: str = "SWAP",
) -> None:
    """Plot observed versus simulated values for one soil layer."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 6.4))

    observed = matched["observed"]
    simulated = matched["simulated"]

    data_min = min(
        observed.min(),
        simulated.min(),
    )

    data_max = max(
        observed.max(),
        simulated.max(),
    )

    margin = 0.01

    axis_min = data_min - margin
    axis_max = data_max + margin

    ax.scatter(
        observed,
        simulated,
        s=65,
        zorder=3,
        label="Matched observations",
    )

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        linewidth=1.5,
        label="1:1 line",
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.set_xlabel(r"Observed volumetric water content (cm$^3$ cm$^{-3}$)")
    ax.set_ylabel(r"Simulated volumetric water content (cm$^3$ cm$^{-3}$)")

    ax.set_title(
        f"{title_prefix} SWAP Performance "
        f"Against Field Observations ({layer.name} cm Soil Layer)"
    )

    metrics_text = (
        f"$R^2$ (Pearson$^2$) = {metrics.r2:.6f}\n"
        f"NSE = {metrics.nse:.6f}\n"
        f"RMSE = {metrics.rmse:.6f}\n"
        f"MAE = {metrics.mae:.6f}\n"
        f"MBE = {metrics.mbe:.6f}\n"
        f"n = {metrics.n}"
    )

    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.9,
        },
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )

    ax.legend(
        frameon=True,
        loc="lower right",
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
