from pathlib import Path

import pytest

from calibration import CalibrationRunner  # type: ignore
from parameters import SoilParameters  # type: ignore
from swap import SwapModel  # type: ignore

MODEL_DIR = Path("model")


@pytest.mark.integration
def test_single_parameter_set_runs_end_to_end(tmp_path):
    observation_path = tmp_path / "observations.csv"

    observation_path.write_text(
        "date,value\n" "2018-10-23,0.319\n" "2018-10-31,0.320\n" "2018-11-08,0.330\n",
        encoding="utf-8",
    )

    parameters = SoilParameters(
        ores=0.0875658533777091,
        osat=0.362417255147982,
        alfa=0.00603938365321607,
        npar=1.10657211418239,
        ksatfit=19.6334210990903,
        lexp=0.565929999579225,
        ksatexm=27.4823857626600,
        bdens=1428.27094863731,
    )

    runner = CalibrationRunner(
        model_dir=MODEL_DIR,
        observation_path=observation_path,
    )

    result = runner.run(parameters)

    assert result.n_observations == 3
    assert result.n_simulated == 248
    assert result.n_matched == 3
    assert result.n_unmatched == 0

    assert not result.simulation.empty
    assert not result.matched.empty

    assert list(result.matched.columns) == [
        "date",
        "observed",
        "simulated",
    ]


@pytest.mark.integration
def test_real_calibration_pipeline_in_isolated_workspace(tmp_path):
    source_model_dir = Path("model")
    observation_path = source_model_dir / "observation.csv"
    trial_dir = tmp_path / "calibration_trial"

    parameters = SoilParameters(
        ores=0.0875658533777091,
        osat=0.362417255147982,
        alfa=0.00603938365321607,
        npar=1.10657211418239,
        ksatfit=19.6334210990903,
        lexp=0.565929999579225,
        ksatexm=27.4823857626600,
        bdens=1428.27094863731,
    )

    model = SwapModel.create_trial_workspace(
        source_model_dir=source_model_dir,
        trial_dir=trial_dir,
    )

    runner = CalibrationRunner(
        model_dir=trial_dir,
        observation_path=observation_path,
        observation_layer=(0.0, 5.0),
    )

    result = runner.run(parameters)

    assert result.n_observations == 10
    assert result.n_simulated == 248
    assert result.n_matched == 10
    assert result.n_unmatched == 0

    assert result.metrics.n == 10

    assert result.metrics.rmse == pytest.approx(
        0.027357,
        abs=1e-6,
    )

    assert result.metrics.mae == pytest.approx(
        0.021560,
        abs=1e-6,
    )

    assert result.metrics.mbe == pytest.approx(
        -0.013840,
        abs=1e-6,
    )

    assert result.metrics.r2 == pytest.approx(
        0.485691,
        abs=1e-6,
    )

    assert result.metrics.nse == pytest.approx(
        -0.015290,
        abs=1e-6,
    )

    assert model.vap_path.exists()
    assert (trial_dir / "swap.ok").exists()
    assert (trial_dir / "swap_swap.log").exists()


@pytest.mark.integration
def test_optuna_objective_runs_real_trial(tmp_path):
    import optuna

    from objective import SwapCalibrationObjective  # type: ignore

    source_model_dir = Path("model")
    observation_path = source_model_dir / "observation.csv"
    runs_dir = tmp_path / "optuna_runs"

    objective = SwapCalibrationObjective(
        source_model_dir=source_model_dir,
        observation_path=observation_path,
        runs_dir=runs_dir,
    )

    study = optuna.create_study(
        direction="minimize",
    )

    study.optimize(
        objective,
        n_trials=1,
    )

    assert len(study.trials) == 1

    trial = study.trials[0]

    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert trial.value is not None
    assert trial.value >= 0.0

    assert trial.user_attrs["n_matched"] == 10

    trial_dir = runs_dir / "trial_0000"

    assert trial_dir.exists()
    assert (trial_dir / "SWAP.swp").exists()
    assert (trial_dir / "swap.ok").exists()
    assert (trial_dir / "swap_swap.log").exists()
    assert (trial_dir / "results" / "result.vap").exists()


@pytest.mark.integration
def test_optuna_objective_reproduces_matlab_baseline(tmp_path):
    import optuna

    from objective import SwapCalibrationObjective  # type: ignore
    from parameters import SoilParameters  # type: ignore

    source_model_dir = Path("model")
    observation_path = source_model_dir / "observation.csv"
    runs_dir = tmp_path / "optuna_runs"

    objective = SwapCalibrationObjective(
        source_model_dir=source_model_dir,
        observation_path=observation_path,
        runs_dir=runs_dir,
    )

    parameters = SoilParameters(
        ores=0.0875658533777091,
        osat=0.362417255147982,
        alfa=0.00603938365321607,
        npar=1.10657211418239,
        ksatfit=19.6334210990903,
        lexp=0.565929999579225,
        ksatexm=27.4823857626600,
        bdens=1428.27094863731,
    )

    study = optuna.create_study(
        direction="minimize",
    )

    trial = study.ask()

    rmse = objective.evaluate_parameters(
        trial=trial,
        parameters=parameters,
    )

    study.tell(
        trial,
        rmse,
    )

    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE

    assert rmse == pytest.approx(
        0.027357,
        abs=1e-6,
    )

    assert trial.user_attrs["mae"] == pytest.approx(
        0.021560,
        abs=1e-6,
    )

    assert trial.user_attrs["mbe"] == pytest.approx(
        -0.013840,
        abs=1e-6,
    )

    assert trial.user_attrs["r2"] == pytest.approx(
        0.485691,
        abs=1e-6,
    )

    assert trial.user_attrs["nse"] == pytest.approx(
        -0.015290,
        abs=1e-6,
    )

    assert trial.user_attrs["n_matched"] == 10
