from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VAP_COLUMNS = {
    "date",
    "depth",
    "wcontent",
    "phead",
    "hconduc",
    "drainage",
    "rootext",
    "waterflux",
    "temp",
    "solute1",
    "solute2",
    "soluteflux",
    "top",
    "bottom",
    "day",
    "dcum",
}


DEFAULT_AGGREGATION = {
    "wcontent": "weighted_mean",
    "phead": "weighted_mean",
    "hconduc": "weighted_mean",
    "temp": "weighted_mean",
    "solute1": "weighted_mean",
    "solute2": "weighted_mean",
    "drainage": "top",
    "waterflux": "top",
    "soluteflux": "top",
    "rootext": "sum",
}


@dataclass(frozen=True)
class RequestedLayer:
    """Requested soil layer in centimetres, positive downward."""

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
class RequestedColumn:
    """VAP variable and aggregation method requested by the user."""

    name: str
    aggregation: str | None = None

    def method(self) -> str:
        if self.name not in VAP_COLUMNS:
            raise ValueError(f"Unknown VAP column: {self.name}")

        aggregation = self.aggregation

        if aggregation is None:
            aggregation = DEFAULT_AGGREGATION.get(self.name)

        if aggregation not in {
            "weighted_mean",
            "mean",
            "sum",
            "top",
            "bottom",
        }:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        return aggregation


class VAPReader:
    """Read and extract depth-based data from a SWAP VAP file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def read_raw(self) -> pd.DataFrame:
        """Read actual soil compartments from the VAP file."""

        rows = []

        for line in self.path.read_text(encoding="latin-1").splitlines():

            line = line.strip()

            if not line:
                continue

            if line.startswith("*"):
                continue

            if line.startswith("date,"):
                continue

            if line.startswith("cm,"):
                continue

            parts = [part.strip() for part in line.split(",")]

            if len(parts) != 16:
                continue

            if parts[0].lower() == "date":
                continue

            try:
                date = pd.to_datetime(parts[0])
                depth = float(parts[1])
                top = float(parts[12])
                bottom = float(parts[13])
            except ValueError:
                continue

            row = {
                "date": date,
                "depth": depth,
                "wcontent": self._to_float(parts[2]),
                "phead": self._to_float(parts[3]),
                "hconduc": self._to_float(parts[4]),
                "drainage": self._to_float(parts[5]),
                "rootext": self._to_float(parts[6]),
                "waterflux": self._to_float(parts[7]),
                "temp": self._to_float(parts[8]),
                "solute1": self._to_float(parts[9]),
                "solute2": self._to_float(parts[10]),
                "soluteflux": self._to_float(parts[11]),
                "top": top,
                "bottom": bottom,
                "day": self._to_float(parts[14]),
                "dcum": self._to_float(parts[15]),
            }

            # Boundary rows have no compartment thickness.
            if bottom >= top:
                continue

            rows.append(row)

        if not rows:
            raise ValueError(f"No valid soil compartments found in {self.path}")

        return pd.DataFrame(rows)

    @staticmethod
    def _to_float(value: str) -> float | None:
        if not value:
            return None

        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _overlap(
        compartment_top: float,
        compartment_bottom: float,
        layer: RequestedLayer,
    ) -> float:
        """Return overlap thickness between a compartment and a layer."""

        compartment_top_cm = -compartment_top
        compartment_bottom_cm = -compartment_bottom

        overlap_top = max(layer.top, compartment_top_cm)
        overlap_bottom = min(layer.bottom, compartment_bottom_cm)

        return max(0.0, overlap_bottom - overlap_top)

    def extract(
        self,
        layers: list[RequestedLayer],
        columns: list[RequestedColumn | str],
    ) -> pd.DataFrame:
        """Extract requested soil layers and variables."""

        raw = self.read_raw()

        requested_columns = [
            column if isinstance(column, RequestedColumn) else RequestedColumn(column)
            for column in columns
        ]

        for column in requested_columns:
            column.method()

        records = []

        for (date,), daily in raw.groupby(["date"], sort=True):
            for layer in layers:
                record = {
                    "date": date,
                    "layer": layer.name,
                }

                for column in requested_columns:
                    record[column.name] = self._aggregate(
                        daily,
                        layer,
                        column,
                    )

                records.append(record)

        return pd.DataFrame(records)

    def _aggregate(
        self,
        daily: pd.DataFrame,
        layer: RequestedLayer,
        column: RequestedColumn,
    ) -> float | None:
        """Aggregate one variable over one requested layer."""

        method = column.method()

        overlaps = daily.apply(
            lambda row: self._overlap(
                row["top"],
                row["bottom"],
                layer,
            ),
            axis=1,
        )

        selected = daily.loc[overlaps > 0].copy()
        selected["overlap"] = overlaps[overlaps > 0]

        if selected.empty:
            return None

        values = selected[column.name].dropna()

        if values.empty:
            return None

        if method == "weighted_mean":
            valid = selected[column.name].notna()

            selected = selected.loc[valid]

            if selected.empty:
                return None

            return float(
                (selected[column.name] * selected["overlap"]).sum()
                / selected["overlap"].sum()
            )

        if method == "mean":
            return float(values.mean())

        if method == "sum":
            return float(values.sum())

        if method == "top":
            selected = selected.sort_values("top", ascending=False)
            return float(selected.iloc[0][column.name])

        if method == "bottom":
            selected = selected.sort_values("bottom", ascending=True)
            return float(selected.iloc[0][column.name])

        raise ValueError(f"Unsupported aggregation method: {method}")
