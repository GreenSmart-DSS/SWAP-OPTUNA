from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ObservationLayer:
    """Observed soil layer in centimeters, positiv downward."""

    top: float
    bottom: float

    def __post_init__(self):
        if self.top < 0:
            raise ValueError("Layer top must be non-negative.")

        if self.bottom <= self.top:
            raise ValueError("Layer bottom must be greater than layer top.")

    @property
    def name(self) -> str:
        return f"{self.top:g}-{self.bottom:g}"


@dataclass(frozen=True)
class ObservationVariable:
    """Definition of an observed variable."""

    name: str
    unit: str


class ObservationReader:
    """Read and validate field observations."""

    REQUITED_COLUMNS = {"date", "value"}  # noqa: RUF012

    def __init__(
        self, path: Path, variable: ObservationVariable, layer: ObservationLayer
    ):
        self.path = Path(path)
        self.variable = variable
        self.layer = layer

    def read(self) -> pd.DataFrame:
        """Read observation and return validated records."""

        data = pd.read_csv(self.path)

        missing = self.REQUITED_COLUMNS - set(data.columns)

        if missing:
            raise ValueError(f"Missing required observation columns: {sorted(missing)}")

        data = data.copy()

        data["date"] = pd.to_datetime(data["date"], errors="coerce")

        if data["date"].isna().any():
            raise ValueError("Observation data contains invalid dates.")

        data["value"] = pd.to_numeric(data["value"], errors="coerce")

        if data["value"].isna().any():
            raise ValueError("Observation data contains invalid values.")

        if data["date"].duplicated().any():
            raise ValueError("Observation data contains duplicate dates.")

        data = data.sort_values("date").reset_index(drop=True)

        return data[["date", "value"]]
