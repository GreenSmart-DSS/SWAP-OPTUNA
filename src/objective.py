from pathlib import Path

import optuna

from calibration import CalibrationRunner
from parameters import PARAMETER_BOUNDS, SoilParameters
from swap import SwapModel


class SwapCalibrationObjective:
    """Optuna objective for SWAP soil hydraulic parameter calibration."""

    def __init__(
        self,
        source_model_dir: Path,
        observation_path: Path,
        runs_dir: Path,
        observation_layer: tuple[float, float] = (0.0, 5.0),
    ):
        self.source_model_dir = Path(source_model_dir)
        self.observation_path = Path(observation_path)
        self.runs_dir = Path(runs_dir)
        self.observation_layer = observation_layer

    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate one Optuna trial and return RMSE."""

        parameters = self._suggest_parameters(trial)

        return self.evaluate_parameters(
            trial=trial,
            parameters=parameters,
        )

    def evaluate_parameters(
        self,
        trial: optuna.Trial,
        parameters: SoilParameters,
    ) -> float:
        """Evaluate a specific soil parameter set."""

        trial_dir = self.runs_dir / f"trial_{trial.number:04d}"

        model = SwapModel.create_trial_workspace(
            source_model_dir=self.source_model_dir,
            trial_dir=trial_dir,
        )

        runner = CalibrationRunner(
            model_dir=model.model_dir,
            observation_path=self.observation_path,
            observation_layer=self.observation_layer,
        )

        result = runner.run(parameters)

        trial.set_user_attr(
            "mae",
            result.metrics.mae,
        )
        trial.set_user_attr(
            "mbe",
            result.metrics.mbe,
        )
        trial.set_user_attr(
            "r2",
            result.metrics.r2,
        )
        trial.set_user_attr(
            "nse",
            result.metrics.nse,
        )
        trial.set_user_attr(
            "n_matched",
            result.n_matched,
        )

        return result.metrics.rmse

    @staticmethod
    def _suggest_parameters(
        trial: optuna.Trial,
    ) -> SoilParameters:
        """Suggest one soil parameter set within the legacy bounds."""

        values = {}

        for name, (lower, upper) in PARAMETER_BOUNDS.items():
            values[name] = trial.suggest_float(
                name,
                lower,
                upper,
            )

        return SoilParameters(**values)
