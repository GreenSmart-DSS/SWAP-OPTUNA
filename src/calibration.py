from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from matching import ObservationMatcher
from metrics import MetricsCalculator, RegressionMetrics
from observations import (
    ObservationLayer,
    ObservationReader,
    ObservationVariable,
)
from parameters import SoilParameters
from swap import SwapModel
from vap import RequestedColumn, RequestedLayer, VAPReader


@dataclass(frozen=True)
class CalibrationResult:
    """Result of evaluating one soil parameter set."""

    parameters: SoilParameters
    observations: pd.DataFrame
    simulation: pd.DataFrame
    matched: pd.DataFrame
    metrics: RegressionMetrics

    @property
    def n_observations(self) -> int:
        """Return the total number of field observations."""
        return len(self.observations)

    @property
    def n_simulated(self) -> int:
        """Return the number of simulated dates."""
        return len(self.simulation)

    @property
    def n_matched(self) -> int:
        """Return the number of exact observation-simulation matches."""
        return len(self.matched)

    @property
    def n_unmatched(self) -> int:
        """Return the number of observations without an exact match."""
        return self.n_observations - self.n_matched


class CalibrationRunner:
    """Run and evaluate SWAP for one soil parameter set."""

    def __init__(
        self,
        model_dir: Path,
        observation_path: Path,
        observation_layer: tuple[float, float] = (0.0, 5.0),
    ):
        self.model = SwapModel(model_dir)
        self.observation_path = Path(observation_path)
        self.observation_layer = observation_layer

        self.observation_reader = ObservationReader(
            path=self.observation_path,
            variable=ObservationVariable(
                name="wcontent",
                unit="cm3/cm3",
            ),
            layer=ObservationLayer(
                top=observation_layer[0],
                bottom=observation_layer[1],
            ),
        )

        self.matcher = ObservationMatcher()
        self.metrics_calculator = MetricsCalculator()

    def run(
        self,
        parameters: SoilParameters,
    ) -> CalibrationResult:
        """Run SWAP and evaluate the requested soil layer."""

        self.model.run(parameters)

        simulation = self._read_simulation()

        observations = self.observation_reader.read()

        matched = self.matcher.match(
            observations,
            simulation,
        )

        metrics = self.metrics_calculator.calculate(
            matched,
        )

        return CalibrationResult(
            parameters=parameters,
            observations=observations,
            simulation=simulation,
            matched=matched,
            metrics=metrics,
        )

    def _read_simulation(self) -> pd.DataFrame:
        """Read simulated water content for the requested layer."""

        vap_reader = VAPReader(
            self.model.vap_path,
        )

        top, bottom = self.observation_layer

        simulation = vap_reader.extract(
            layers=[
                RequestedLayer(
                    top=top,
                    bottom=bottom,
                )
            ],
            columns=[
                RequestedColumn(
                    name="wcontent",
                    aggregation="weighted_mean",
                )
            ],
        )

        layer_name = f"{top:g}-{bottom:g}"

        simulation = simulation[simulation["layer"] == layer_name].copy()

        simulation = simulation[["date", "wcontent"]].rename(
            columns={"wcontent": "value"}
        )

        return simulation.reset_index(drop=True)
