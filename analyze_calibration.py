from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.importance import (
    FanovaImportanceEvaluator,
    PedAnovaImportanceEvaluator,
    get_param_importances,
)
from optuna.visualization.matplotlib import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STUDY_NAME = "swap_soil_calibration_tpe"
STORAGE = "sqlite:///optuna_study.db"

OUTPUT_DIR = Path("calibration_analysis")
FIGURES_DIR = OUTPUT_DIR / "figures"
SLICES_DIR = FIGURES_DIR / "slices"
CONTOURS_DIR = FIGURES_DIR / "contours"
CONVERGENCE_DIR = FIGURES_DIR / "convergence"
PARAMETERS_DIR = FIGURES_DIR / "parameters"
BASELINE_DIR = FIGURES_DIR / "baseline"
DATA_DIR = OUTPUT_DIR / "data"

TOP_N = 10
TOP_IMPORTANCE_PARAMETERS = 4

PARAMETERS = [
    "ores",
    "osat",
    "alfa",
    "npar",
    "ksatfit",
    "lexp",
    "ksatexm",
    "bdens",
]

PARAMETER_BOUNDS = {
    "ores": (0.06, 0.09),
    "osat": (0.25, 0.50),
    "alfa": (0.006, 0.0082),
    "npar": (1.10, 1.61),
    "ksatfit": (10.01, 20.00),
    "lexp": (0.40, 1.99),
    "ksatexm": (20.01, 40.01),
    "bdens": (1350.01, 1600.01),
}

# Legacy MATLAB calibration reported in the project documentation.
MATLAB_BASELINE = {
    "rmse": 0.027357,
    "mae": 0.021560,
    "mbe": -0.013840,
    "r2": 0.485691,
    "nse": -0.015290,
}

# ---------------------------------------------------------------------
# Study and trial data
# ---------------------------------------------------------------------


def load_study() -> optuna.Study:
    """Load the existing Optuna study."""
    return optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
    )


def get_complete_trials(
    study: optuna.Study,
) -> list[optuna.trial.FrozenTrial]:
    """Return completed trials with valid objective values."""
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def build_trial_dataframe(
    trials: list[optuna.trial.FrozenTrial],
) -> pd.DataFrame:
    """Build a tabular representation of completed trials."""
    rows = []

    for trial in trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "rmse": trial.value,
        }
        row.update(trial.params)
        rows.append(row)

    return pd.DataFrame(rows)


def save_trial_data(df: pd.DataFrame) -> None:
    """Save all trials and the best trials."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        DATA_DIR / "trial_results.csv",
        index=False,
    )

    top_trials = (
        df.sort_values("rmse", ascending=True).head(TOP_N).reset_index(drop=True)
    )

    top_trials.to_csv(
        DATA_DIR / "top_trials.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# Optuna visualizations
# ---------------------------------------------------------------------


def save_optimization_history(study: optuna.Study) -> None:
    """Save the Optuna optimization history."""
    ax = plot_optimization_history(study)

    ax.figure.savefig(  # type: ignore
        FIGURES_DIR / "optimization_history.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(ax.figure)  # type: ignore


def save_parameter_importance_plot(study: optuna.Study) -> None:
    """Save the default Optuna parameter importance plot."""
    ax = plot_param_importances(study)

    ax.figure.savefig(  # type: ignore
        FIGURES_DIR / "parameter_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(ax.figure)  # type: ignore


def save_slice_plots(study: optuna.Study) -> None:
    """Save one slice plot for each calibrated parameter."""
    for parameter in PARAMETERS:
        ax = plot_slice(
            study,
            params=[parameter],
        )

        ax.figure.savefig(  # type: ignore
            SLICES_DIR / f"slice_{parameter}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(ax.figure)  # type: ignore


def save_parallel_coordinate_plot(study: optuna.Study) -> None:
    """Save the Optuna parallel-coordinate visualization."""
    ax = plot_parallel_coordinate(study)

    ax.figure.savefig(  # type: ignore
        FIGURES_DIR / "parallel_coordinate.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(ax.figure)  # type: ignore


# ---------------------------------------------------------------------
# Parameter importance
# ---------------------------------------------------------------------


def calculate_parameter_importance(
    study: optuna.Study,
    evaluator,
) -> dict[str, float]:
    """Calculate parameter importance with a specified evaluator."""
    return get_param_importances(
        study,
        evaluator=evaluator,
    )


def save_parameter_importance(
    study: optuna.Study,
    evaluator,
    filename: str,
) -> dict[str, float]:
    """Calculate and save parameter importance."""
    importances = calculate_parameter_importance(
        study,
        evaluator,
    )

    importance_df = pd.DataFrame(
        {
            "parameter": list(importances.keys()),
            "importance": list(importances.values()),
        }
    )

    importance_df.to_csv(
        DATA_DIR / filename,
        index=False,
    )

    return importances


# ---------------------------------------------------------------------
# Contour analysis
# ---------------------------------------------------------------------


def save_contour_plots(
    study: optuna.Study,
    importance: dict[str, float],
) -> None:
    """Save contour plots for pairs of important parameters."""
    selected = list(importance.keys())[:TOP_IMPORTANCE_PARAMETERS]

    for parameter_x, parameter_y in combinations(selected, 2):
        ax = plot_contour(
            study,
            params=[parameter_x, parameter_y],
        )

        filename = f"contour_{parameter_x}_{parameter_y}.png"

        ax.figure.savefig(  # type: ignore
            CONTOURS_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(ax.figure)  # type: ignore


# ---------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------


def build_convergence_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate best-so-far RMSE and improvement by trial."""
    result = df.sort_values("number").copy()

    result["best_so_far_rmse"] = result["rmse"].cummin()

    result["previous_best_rmse"] = result["best_so_far_rmse"].shift(1)

    result["improvement"] = result["previous_best_rmse"] - result["best_so_far_rmse"]

    result["relative_improvement_percent"] = (
        result["improvement"] / result["previous_best_rmse"] * 100
    )

    return result


def save_convergence_analysis(df: pd.DataFrame) -> None:
    """Save convergence statistics and figures."""
    convergence = build_convergence_dataframe(df)

    convergence.to_csv(
        DATA_DIR / "convergence_statistics.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        convergence["number"],
        convergence["best_so_far_rmse"],
        marker="o",
        markersize=3,
    )

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best-so-far RMSE")
    ax.set_title("Calibration convergence")
    ax.grid(True, alpha=0.25)

    fig.savefig(
        CONVERGENCE_DIR / "best_so_far_rmse.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df["rmse"],
        bins=12,
    )

    ax.set_xlabel("RMSE")
    ax.set_ylabel("Number of trials")
    ax.set_title("Distribution of trial RMSE")

    fig.savefig(
        CONVERGENCE_DIR / "rmse_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.50]

    rows = []

    for quantile in quantiles:
        n = max(1, int(len(df) * quantile))
        selected = df.nsmallest(n, "rmse")

        rows.append(
            {
                "top_fraction": quantile,
                "n_trials": n,
                "rmse_min": selected["rmse"].min(),
                "rmse_median": selected["rmse"].median(),
                "rmse_mean": selected["rmse"].mean(),
                "rmse_max": selected["rmse"].max(),
            }
        )

    pd.DataFrame(rows).to_csv(
        DATA_DIR / "rmse_quantile_summary.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# Parameter boundary analysis
# ---------------------------------------------------------------------


def build_parameter_diagnostics(
    study: optuna.Study,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build parameter boundary, correlation, and top-trial diagnostics."""
    best = study.best_trial

    top10 = df.nsmallest(
        max(1, int(len(df) * 0.10)),
        "rmse",
    )

    top20 = df.nsmallest(
        max(1, int(len(df) * 0.20)),
        "rmse",
    )

    rows = []

    for parameter in PARAMETERS:
        lower, upper = PARAMETER_BOUNDS[parameter]
        best_value = best.params[parameter]

        normalized = (best_value - lower) / (upper - lower)

        distance_lower = best_value - lower
        distance_upper = upper - best_value

        rows.append(
            {
                "parameter": parameter,
                "lower_bound": lower,
                "upper_bound": upper,
                "best_value": best_value,
                "normalized_best_position": normalized,
                "distance_to_lower": distance_lower,
                "distance_to_upper": distance_upper,
                "near_lower_bound": normalized <= 0.05,
                "near_upper_bound": normalized >= 0.95,
                "top10_min": top10[parameter].min(),
                "top10_q25": top10[parameter].quantile(0.25),
                "top10_median": top10[parameter].median(),
                "top10_q75": top10[parameter].quantile(0.75),
                "top10_max": top10[parameter].max(),
                "top10_iqr": (
                    top10[parameter].quantile(0.75) - top10[parameter].quantile(0.25)
                ),
                "top20_min": top20[parameter].min(),
                "top20_median": top20[parameter].median(),
                "top20_max": top20[parameter].max(),
            }
        )

    return pd.DataFrame(rows)


def save_parameter_diagnostics(
    study: optuna.Study,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Save parameter diagnostics and a normalized-position figure."""
    diagnostics = build_parameter_diagnostics(
        study,
        df,
    )

    diagnostics.to_csv(
        DATA_DIR / "parameter_diagnostics.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        diagnostics["parameter"],
        diagnostics["normalized_best_position"],
    )

    ax.axhline(
        0.05,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        0.95,
        linestyle="--",
        linewidth=1,
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized position within parameter bounds")
    ax.set_xlabel("Parameter")
    ax.set_title("Best parameter position within calibration bounds")
    ax.tick_params(axis="x", rotation=45)

    fig.savefig(
        PARAMETERS_DIR / "parameter_positions.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return diagnostics


# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------


def save_correlation_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations with RMSE."""
    rows = []

    for parameter in PARAMETERS:
        rows.append(
            {
                "parameter": parameter,
                "pearson_correlation": df[parameter].corr(df["rmse"], method="pearson"),
                "spearman_correlation": df[parameter].corr(
                    df["rmse"], method="spearman"
                ),
            }
        )

    correlation_df = pd.DataFrame(rows)

    correlation_df.to_csv(
        DATA_DIR / "parameter_correlations.csv",
        index=False,
    )

    plot_df = correlation_df.sort_values(
        "spearman_correlation",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        plot_df["parameter"],
        plot_df["spearman_correlation"],
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Spearman correlation with RMSE")
    ax.set_ylabel("Parameter")
    ax.set_title("Parameter–RMSE rank correlations")

    fig.savefig(
        FIGURES_DIR / "parameter_correlations.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return correlation_df


# ---------------------------------------------------------------------
# Top-trial analysis
# ---------------------------------------------------------------------


def save_top_trial_parameter_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize parameter distributions among the top trials."""
    top_trials = df.nsmallest(
        TOP_N,
        "rmse",
    )

    parameter_data = top_trials[PARAMETERS]

    summary = pd.DataFrame(
        {
            "min": parameter_data.min(),
            "q1": parameter_data.quantile(0.25),
            "median": parameter_data.median(),
            "q3": parameter_data.quantile(0.75),
            "max": parameter_data.max(),
            "mean": parameter_data.mean(),
            "std": parameter_data.std(),
        }
    )

    summary["iqr"] = summary["q3"] - summary["q1"]

    summary.index.name = "parameter"

    summary.to_csv(
        DATA_DIR / "top_trial_parameter_summary.csv",
    )

    return summary


def save_top_trial_distributions(
    df: pd.DataFrame,
) -> None:
    """Save parameter distribution plots for the top trials."""
    top_trials = df.nsmallest(
        TOP_N,
        "rmse",
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 8),
    )

    for ax, parameter in zip(
        axes.flat,
        PARAMETERS,
    ):
        ax.boxplot(
            top_trials[parameter].dropna(),
            vertical=True,
        )

        ax.set_title(parameter)
        ax.set_xticks([])

    fig.suptitle(
        f"Parameter distributions among top-{TOP_N} trials",
    )

    fig.tight_layout()

    fig.savefig(
        PARAMETERS_DIR / "top_trial_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Parameter-pair analysis
# ---------------------------------------------------------------------


def save_parameter_pair_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare parameter correlations for all and top trials."""
    top10 = df.nsmallest(
        max(1, int(len(df) * 0.10)),
        "rmse",
    )

    top20 = df.nsmallest(
        max(1, int(len(df) * 0.20)),
        "rmse",
    )

    rows = []

    for parameter_x, parameter_y in combinations(
        PARAMETERS,
        2,
    ):
        rows.append(
            {
                "parameter_1": parameter_x,
                "parameter_2": parameter_y,
                "all_trial_pearson": df[parameter_x].corr(
                    df[parameter_y],
                    method="pearson",
                ),
                "top10_pearson": top10[parameter_x].corr(
                    top10[parameter_y],
                    method="pearson",
                ),
                "top20_pearson": top20[parameter_x].corr(
                    top20[parameter_y],
                    method="pearson",
                ),
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        DATA_DIR / "parameter_pair_correlations.csv",
        index=False,
    )

    return result


# ---------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------


def save_baseline_comparison(
    study: optuna.Study,
) -> pd.DataFrame:
    """Compare Optuna best metrics with the legacy MATLAB baseline."""
    best_trial = study.best_trial

    optuna_metrics = {
        "rmse": best_trial.value,
        "mae": best_trial.user_attrs.get("mae"),
        "mbe": best_trial.user_attrs.get("mbe"),
        "r2": best_trial.user_attrs.get("r2"),
        "nse": best_trial.user_attrs.get("nse"),
    }

    rows = []

    for metric in MATLAB_BASELINE:  # noqa: PLC0206
        matlab_value = MATLAB_BASELINE[metric]
        optuna_value = optuna_metrics[metric]

        if optuna_value is None:
            difference = None
            relative_difference = None
        else:
            difference = optuna_value - matlab_value

            if matlab_value != 0:
                relative_difference = difference / abs(matlab_value) * 100
            else:
                relative_difference = None

        rows.append(
            {
                "metric": metric,
                "matlab_baseline": matlab_value,
                "optuna_best": optuna_value,
                "difference_optuna_minus_matlab": difference,
                "relative_difference_percent": relative_difference,
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        DATA_DIR / "baseline_comparison.csv",
        index=False,
    )

    plot_df = result.dropna(
        subset=["optuna_best"],
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    x = range(len(plot_df))

    width = 0.35

    ax.bar(
        [value - width / 2 for value in x],
        plot_df["matlab_baseline"],
        width=width,
        label="MATLAB baseline",
    )

    ax.bar(
        [value + width / 2 for value in x],
        plot_df["optuna_best"],
        width=width,
        label="Optuna best",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        plot_df["metric"],
    )

    ax.set_ylabel("Metric value")
    ax.set_title("MATLAB baseline and Optuna best")
    ax.legend()

    fig.savefig(
        BASELINE_DIR / "baseline_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return result


# ---------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------


def create_summary(
    study: optuna.Study,
    trials: list[optuna.trial.FrozenTrial],
    pedanova_importance: dict[str, float],
    fanova_importance: dict[str, float],
    diagnostics: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    """Write a reproducible text summary of the analysis."""
    best_trial = study.best_trial

    lines = [
        "SWAP-OPTUNA Calibration Analysis",
        "================================",
        "",
        "Study",
        "-----",
        f"Name: {study.study_name}",
        f"Direction: {study.direction.name}",
        f"Total trials: {len(study.trials)}",
        f"Completed trials: {len(trials)}",
        "",
        "Best trial",
        "----------",
        f"Trial number: {best_trial.number}",
        f"Best RMSE: {best_trial.value:.10f}",
        "",
        "Best parameters",
        "---------------",
    ]

    for parameter in PARAMETERS:
        lines.append(f"{parameter}: {best_trial.params[parameter]:.12g}")

    lines.extend(
        [
            "",
            "Best-trial metrics",
            "------------------",
            f"RMSE: {best_trial.value:.10f}",
            f"MAE: {best_trial.user_attrs.get('mae')}",
            f"MBE: {best_trial.user_attrs.get('mbe')}",
            f"R2: {best_trial.user_attrs.get('r2')}",
            f"NSE: {best_trial.user_attrs.get('nse')}",
            f"Matched observations: "  # noqa: ISC004
            f"{best_trial.user_attrs.get('n_matched')}",
            "",
            "PedANOVA parameter importance",
            "-----------------------------",
        ]
    )

    for parameter, value in pedanova_importance.items():
        lines.append(f"{parameter}: {value:.8f}")

    lines.extend(
        [
            "",
            "fANOVA parameter importance",
            "---------------------------",
        ]
    )

    for parameter, value in fanova_importance.items():
        lines.append(f"{parameter}: {value:.8f}")

    lines.extend(
        [
            "",
            "Boundary diagnostics",
            "--------------------",
        ]
    )

    for _, row in diagnostics.iterrows():
        boundary_note = []

        if row["near_lower_bound"]:
            boundary_note.append("near lower bound")

        if row["near_upper_bound"]:
            boundary_note.append("near upper bound")

        if not boundary_note:
            boundary_note.append("not near a bound")

        lines.append(
            f"{row['parameter']}: "
            f"best={row['best_value']:.12g}; "
            f"normalized_position="
            f"{row['normalized_best_position']:.4f}; "
            f"{', '.join(boundary_note)}"
        )

    lines.extend(
        [
            "",
            "Parameter–RMSE correlations",
            "----------------------------",
        ]
    )

    for _, row in correlations.iterrows():
        lines.append(
            f"{row['parameter']}: "
            f"Pearson={row['pearson_correlation']:.6f}; "
            f"Spearman={row['spearman_correlation']:.6f}"
        )

    lines.extend(
        [
            "",
            "Interpretation notes",
            "--------------------",
            "Parameter importance describes the relationship between "  # noqa: ISC004
            "sampled parameter values and the optimization objective "
            "within this Optuna study.",
            "",
            "fANOVA and PedANOVA use different statistical perspectives "  # noqa: ISC004
            "and may therefore produce different rankings.",
            "",
            "Parameter importance is not a direct measure of physical "  # noqa: ISC004
            "parameter importance in the SWAP model.",
            "",
            "Pearson and Spearman correlations are screening diagnostics "  # noqa: ISC004
            "and should not be interpreted as global sensitivity indices.",
            "",
            "Boundary proximity indicates that the best sampled value "  # noqa: ISC004
            "is close to a predefined calibration bound. It does not "
            "by itself justify changing the bound.",
            "",
            "Top-trial parameter ranges describe empirical distributions "  # noqa: ISC004
            "among the best sampled trials and should not be interpreted "
            "as formal confidence intervals.",
            "",
            "The MATLAB comparison is descriptive. It does not establish "  # noqa: ISC004
            "that either calibration method is statistically superior.",
        ]
    )

    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------


def main() -> None:
    """Run the complete calibration diagnostic analysis."""
    for directory in [
        OUTPUT_DIR,
        FIGURES_DIR,
        SLICES_DIR,
        CONTOURS_DIR,
        CONVERGENCE_DIR,
        PARAMETERS_DIR,
        BASELINE_DIR,
        DATA_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    study = load_study()
    trials = get_complete_trials(study)

    if not trials:
        raise RuntimeError("The study contains no completed trials.")

    df = build_trial_dataframe(trials)

    print(f"Study: {study.study_name}")
    print(f"Trials: {len(study.trials)}")
    print(f"Completed trials: {len(trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.10f}")

    # Trial data
    save_trial_data(df)

    # Optuna visualizations
    save_optimization_history(study)
    save_parameter_importance_plot(study)
    save_slice_plots(study)
    save_parallel_coordinate_plot(study)

    # Importance analysis
    pedanova_importance = save_parameter_importance(
        study,
        PedAnovaImportanceEvaluator(),
        "parameter_importance_pedanova.csv",
    )

    fanova_importance = save_parameter_importance(
        study,
        FanovaImportanceEvaluator(),
        "parameter_importance_fanova.csv",
    )

    # Contours based on PedANOVA ranking
    save_contour_plots(
        study,
        pedanova_importance,
    )

    # Convergence
    save_convergence_analysis(df)

    # Parameter diagnostics
    diagnostics = save_parameter_diagnostics(
        study,
        df,
    )

    # Correlations
    correlations = save_correlation_analysis(df)

    # Top-trial analysis
    save_top_trial_parameter_summary(df)
    save_top_trial_distributions(df)

    # Parameter-pair diagnostics
    save_parameter_pair_correlations(df)

    # MATLAB comparison
    save_baseline_comparison(study)

    # Reproducible text summary
    create_summary(
        study,
        trials,
        pedanova_importance,
        fanova_importance,
        diagnostics,
        correlations,
    )

    print(f"Analysis outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
