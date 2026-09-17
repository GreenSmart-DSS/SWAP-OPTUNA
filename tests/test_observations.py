from pathlib import Path

import pandas as pd
import pytest

from observations import (  # type: ignore
    ObservationLayer,
    ObservationReader,
    ObservationVariable,
)


def write_test_observations(path: Path) -> None:
    content = """\
date,value
2018-10-23,0.318
2018-10-31,0.305
2018-11-08,0.291
2018-11-21,0.327
"""

    path.write_text(content, encoding="utf-8")


def test_observation_layer():
    layer = ObservationLayer(top=0, bottom=5)

    assert layer.name == "0-5"


def test_observation_layer_validation():
    with pytest.raises(ValueError):
        ObservationLayer(top=-1, bottom=5)

    with pytest.raises(ValueError):
        ObservationLayer(top=5, bottom=5)


def test_read_observations(tmp_path: Path):
    observation_path = tmp_path / "observations.csv"
    write_test_observations(observation_path)

    reader = ObservationReader(
        path=observation_path,
        variable=ObservationVariable(
            name="wcontent",
            unit="cm3/cm3",
        ),
        layer=ObservationLayer(
            top=0,
            bottom=5,
        ),
    )

    data = reader.read()

    assert len(data) == 4
    assert list(data.columns) == ["date", "value"]

    assert pd.api.types.is_datetime64_any_dtype(data["date"])

    assert data.iloc[0]["date"] == pd.Timestamp("2018-10-23")
    assert data.iloc[0]["value"] == pytest.approx(0.318)


def test_observation_dates_are_not_interpolated(tmp_path: Path):
    observation_path = tmp_path / "observations.csv"
    write_test_observations(observation_path)

    reader = ObservationReader(
        path=observation_path,
        variable=ObservationVariable(
            name="wcontent",
            unit="cm3/cm3",
        ),
        layer=ObservationLayer(
            top=0,
            bottom=5,
        ),
    )

    data = reader.read()

    assert len(data) == 4
    assert pd.Timestamp("2018-10-24") not in set(data["date"])


def test_duplicate_dates_are_rejected(tmp_path: Path):
    observation_path = tmp_path / "observations.csv"

    observation_path.write_text(
        """\
date,value
2018-10-23,0.318
2018-10-23,0.320
""",
        encoding="utf-8",
    )

    reader = ObservationReader(
        path=observation_path,
        variable=ObservationVariable(
            name="wcontent",
            unit="cm3/cm3",
        ),
        layer=ObservationLayer(
            top=0,
            bottom=5,
        ),
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        reader.read()


def test_invalid_value_is_rejected(tmp_path: Path):
    observation_path = tmp_path / "observations.csv"

    observation_path.write_text(
        """\
date,value
2018-10-23,0.318
2018-10-31,invalid
""",
        encoding="utf-8",
    )

    reader = ObservationReader(
        path=observation_path,
        variable=ObservationVariable(
            name="wcontent",
            unit="cm3/cm3",
        ),
        layer=ObservationLayer(
            top=0,
            bottom=5,
        ),
    )

    with pytest.raises(ValueError, match="invalid values"):
        reader.read()
