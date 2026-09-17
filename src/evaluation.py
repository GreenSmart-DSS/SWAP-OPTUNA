from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from matching import ObservationMatcher
from metrics import MetricsCalculator, RegressionMetrics
from observations import ObservationLayer, ObservationReader, ObservationVariable
from vap import RequestedColumn, RequestedLayer, VAPReader


@dataclass(frozen=True)
class LayerEvaluationResult:
    """Evaluation result for one requested soil layer."""

    layer: RequestedLayer
    simulation: pd.DataFrame
    observations: pd.DataFrame | None
    matched: pd.DataFrame | None
    metrics: RegressionMetrics | None

    @property
    def has_observations(self) -> bool:
        """Return whether field observations are available."""
        return self.observations is not None

    @property
    def has_metrics(self) -> bool:
        """Return whether statistical metrics are available."""
        return self.metrics is not None


class OutputEvaluator:
    """Evaluate SWAP output for multiple requested soil layers."""

    def __init__(self):
        self.matcher = ObservationMatcher()
        self.metrics_calculator = MetricsCalculator()

    def evaluate(
        self,
        vap_path: Path,
        layers: list[RequestedLayer],
        observation_paths: dict[str, Path] | None = None,
        columns: list[RequestedColumn] | None = None,
    ) -> list[LayerEvaluationResult]:
        """Evaluate requested SWAP layers against optional observations."""

        if not layers:
            raise ValueError("At least one soil layer must be requested.")

        vap_reader = VAPReader(vap_path)

        columns = columns or [
            RequestedColumn(name="wcontent", aggregation="weighted_mean")
        ]

        simulation = vap_reader.extract(
            layers=layers,
            columns=columns,  # type: ignore
        )

        observation_paths = observation_paths or {}

        results = []

        for layer in layers:
            layer_name = layer.name

            layer_simulation = simulation[simulation["layer"] == layer_name].copy()

            layer_simulation = layer_simulation.drop(columns=["layer"]).reset_index(
                drop=True
            )

            observation_path = observation_paths.get(layer_name)

            if observation_path is None:
                results.append(
                    LayerEvaluationResult(
                        layer=layer,
                        simulation=layer_simulation,
                        observations=None,
                        matched=None,
                        metrics=None,
                    )
                )
                continue

            observations = self._read_observations(observation_path, layer)

            simulation_for_matching = layer_simulation[["date", "wcontent"]].rename(
                columns={"wcontent": "value"}
            )

            matched = self.matcher.match(
                observations,
                simulation_for_matching,
            )

            if matched.empty:
                metrics = None
            else:
                metrics = self.metrics_calculator.calculate(matched)

            results.append(
                LayerEvaluationResult(
                    layer=layer,
                    simulation=layer_simulation,
                    observations=observations,
                    matched=matched,
                    metrics=metrics,
                )
            )
        return results

    @staticmethod
    def _read_observations(
        path: Path,
        layer: RequestedLayer,
    ) -> pd.DataFrame:
        """Read observations associated with one soil layer."""

        reader = ObservationReader(
            path=path,
            variable=ObservationVariable(name="wcontent", unit="cm3/cm3"),
            layer=ObservationLayer(top=layer.top, bottom=layer.bottom),
        )

        return reader.read()
