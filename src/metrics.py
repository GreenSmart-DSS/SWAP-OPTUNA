import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    """Statistical metrics for observed and simulated values."""

    rmse: float
    mae: float
    mbe: float
    r2: float
    nse: float
    n: int


class MetricsCalculator:
    """Calculate statistical metrics from matched observations."""

    def calculate(
        self,
        matched: pd.DataFrame,
    ) -> RegressionMetrics:
        """Calculate metrics using observed and simulated values."""

        self._validate_input(matched)

        observed = matched["observed"].to_numpy(dtype=float)
        simulated = matched["simulated"].to_numpy(dtype=float)

        errors = simulated - observed

        n = len(observed)

        # Root Mean Square Error
        rmse = math.sqrt((errors**2).mean())

        # Mean Absolute Error
        mae = abs(errors).mean()

        # Mean Bias Error
        # Positive values indicate overestimation.
        # Negative values indicate underestimation.
        mbe = errors.mean()

        # Nash-Sutcliffe Efficiency
        observed_mean = observed.mean()

        ss_res = ((observed - simulated) ** 2).sum()
        ss_tot = ((observed - observed_mean) ** 2).sum()

        if ss_tot == 0:
            nse = float("nan")
        else:
            nse = 1.0 - (ss_res / ss_tot)

        # R-squared based on the squared Pearson correlation coefficient
        observed_centered = observed - observed.mean()
        simulated_centered = simulated - simulated.mean()

        denominator = (observed_centered**2).sum() * (simulated_centered**2).sum()

        if denominator == 0:
            r2 = float("nan")
        else:
            r = (observed_centered * simulated_centered).sum() / math.sqrt(denominator)
            r2 = r**2

        return RegressionMetrics(
            rmse=float(rmse),
            mae=float(mae),
            mbe=float(mbe),
            r2=float(r2),
            nse=float(nse),
            n=n,
        )

    @staticmethod
    def _validate_input(
        matched: pd.DataFrame,
    ) -> None:
        """Validate the matched observation structure."""

        required_columns = {
            "date",
            "observed",
            "simulated",
        }

        missing = required_columns - set(matched.columns)

        if missing:
            raise ValueError(
                f"Matched data is missing required columns: " f"{sorted(missing)}"
            )

        if matched.empty:
            raise ValueError("Cannot calculate metrics from empty matched data.")

        if matched["observed"].isna().any():
            raise ValueError("Matched data contains missing observed values.")

        if matched["simulated"].isna().any():
            raise ValueError("Matched data contains missing simulated values.")
