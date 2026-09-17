from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MatchedObservation:
    """A matched observed and simulated value on the same date."""

    date: pd.Timestamp
    observed: float
    simulated: float


class ObservationMatcher:
    """Match field observations with simulated values by exact date."""

    def match(
        self,
        observations: pd.DataFrame,
        simulations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return only observations with an exact simulation date."""

        self._validate_input(observations, "observations")
        self._validate_input(simulations, "simulations")

        observed = observations[["date", "value"]].copy()
        simulated = simulations[["date", "value"]].copy()

        observed = observed.rename(columns={"value": "observed"})
        simulated = simulated.rename(columns={"value": "simulated"})

        matched = observed.merge(
            simulated,
            on="date",
            how="inner",
            validate="one_to_one",
        )

        return matched.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _validate_input(
        data: pd.DataFrame,
        name: str,
    ) -> None:
        """Validate the minimum structure required for matching."""

        required_columns = {"date", "value"}

        missing = required_columns - set(data.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: " f"{sorted(missing)}"
            )

        if data["date"].isna().any():
            raise ValueError(f"{name} contains missing dates.")

        if data["value"].isna().any():
            raise ValueError(f"{name} contains missing values.")

        if data["date"].duplicated().any():
            raise ValueError(f"{name} contains duplicate dates.")
