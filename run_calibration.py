import sys
from pathlib import Path

import optuna

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration import CalibrationRunner  # type: ignore
from objective import SwapCalibrationObjective  # type: ignore
from parameters import SoilParameters  # type: ignore
from plotting import plot_scatter, plot_time_series  # type: ignore
from swap import SwapModel  # type: ignore

STUDY_NAME = "swap_soil_calibration_tpe"
N_TRIALS = 150
DATABASE_PATH = PROJECT_DIR / "optuna_study.db"


def main() -> None:
    """Run Optuna calibration and evaluate the best parameter set."""

    source_model_dir = PROJECT_DIR / "model"
    observation_path = source_model_dir / "observation.csv"
    runs_dir = PROJECT_DIR / "optuna_runs"
    best_results_dir = PROJECT_DIR / "best_results"

    objective = SwapCalibrationObjective(
        source_model_dir=source_model_dir,
        observation_path=observation_path,
        runs_dir=runs_dir,
    )

    sampler = optuna.samplers.TPESampler(seed=42)

    storage = f"sqlite:///{DATABASE_PATH.as_posix()}"

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    best_trial = study.best_trial

    best_parameters = SoilParameters(
        ores=best_trial.params["ores"],
        osat=best_trial.params["osat"],
        alfa=best_trial.params["alfa"],
        npar=best_trial.params["npar"],
        ksatfit=best_trial.params["ksatfit"],
        lexp=best_trial.params["lexp"],
        ksatexm=best_trial.params["ksatexm"],
        bdens=best_trial.params["bdens"],
    )

    best_trial_dir = runs_dir / f"trial_{best_trial.number:04d}_final"

    if best_trial_dir.exists():
        import shutil

        shutil.rmtree(best_trial_dir)

    best_model = SwapModel.create_trial_workspace(
        source_model_dir=source_model_dir,
        trial_dir=best_trial_dir,
    )

    runner = CalibrationRunner(
        model_dir=best_model.model_dir,
        observation_path=observation_path,
    )

    result = runner.run(best_parameters)

    best_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_time_series(
        simulation=result.simulation,
        observations=result.observations,
        metrics=result.metrics,
        output_path=best_results_dir / "best_time_series.png",
        title_prefix="Best Trial",
    )

    plot_scatter(
        matched=result.matched,
        metrics=result.metrics,
        output_path=best_results_dir / "best_scatter.png",
        title_prefix="Best Trial",
    )

    print("\n=== Best Trial ===")
    print(f"Trial: {best_trial.number}")
    print(f"RMSE:  {result.metrics.rmse:.6f}")
    print(f"MAE:   {result.metrics.mae:.6f}")
    print(f"MBE:   {result.metrics.mbe:.6f}")
    print(f"R2:    {result.metrics.r2:.6f}")
    print(f"NSE:   {result.metrics.nse:.6f}")
    print(f"n:     {result.n_matched}")

    print("Parameters:")

    for name, value in best_trial.params.items():
        print(f"  {name}: {value:.12g}")

    print("\nFigures:")
    print(f"  {best_results_dir / 'best_time_series.png'}")
    print(f"  {best_results_dir / 'best_scatter.png'}")


if __name__ == "__main__":
    main()
