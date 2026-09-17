from pathlib import Path

import pandas as pd
import pytest

from vap import RequestedColumn, RequestedLayer, VAPReader  # type: ignore


def write_test_vap(path: Path) -> None:
    content = """\
* Test SWAP VAP file
                 cm,  cm3/cm3,         cm,       cm/d,       cm/d,       cm/d,       cm/d,     C,     mg/cm3,     mg/cm3,   mg/cm2/d,     cm,     cm,  nr,   nr
       date,  depth, wcontent,      phead,    hconduc,    drainage,    rootext,  waterflux,   temp,    solute1,    solute2, soluteflux,    top, bottom, day, dcum
23-Oct-2018,   -0.5,    0.20, -1.000E+02,  1.000E-03,  1.000E-01,  1.000E-02,  2.000E-01,  20.00,  1.000E-01,  2.000E-01,  3.000E-01,    0.0,   -1.0, 296,    0
23-Oct-2018,   -1.5,    0.30, -2.000E+02,  2.000E-03,  2.000E-01,  2.000E-02,  3.000E-01,  21.00,  2.000E-01,  3.000E-01,  4.000E-01,   -1.0,   -2.0, 296,    0
23-Oct-2018,   -2.5,    0.40, -3.000E+02,  3.000E-03,  3.000E-01,  3.000E-02,  4.000E-01,  22.00,  3.000E-01,  4.000E-01,  5.000E-01,   -2.0,   -3.0, 296,    0
23-Oct-2018,   -3.5,    0.50, -4.000E+02,  4.000E-03,  4.000E-01,  4.000E-02,  5.000E-01,  23.00,  4.000E-01,  5.000E-01,  6.000E-01,   -3.0,   -4.0, 296,    0
23-Oct-2018,   -4.5,    0.60, -5.000E+02,  5.000E-03,  5.000E-01,  5.000E-02,  6.000E-01,  24.00,  5.000E-01,  6.000E-01,  7.000E-01,   -4.0,   -5.0, 296,    0
23-Oct-2018,   -5.5,    0.70, -6.000E+02,  6.000E-03,  6.000E-01,  6.000E-02,  7.000E-01,  25.00,  6.000E-01,  7.000E-01,  8.000E-01,   -5.0,  -10.0, 296,    0
23-Oct-2018,  -10.0,                                                     0.0,  -10.0, 296,    0
24-Oct-2018,   -0.5,    0.25, -1.100E+02,  1.100E-03,  1.100E-01,  1.100E-02,  2.100E-01,  20.50,  1.100E-01,  2.100E-01,  3.100E-01,    0.0,   -1.0, 297,    0
24-Oct-2018,   -1.5,    0.35, -2.100E+02,  2.100E-03,  2.100E-01,  2.100E-02,  3.100E-01,  21.50,  2.100E-01,  3.100E-01,  4.100E-01,   -1.0,   -2.0, 297,    0
24-Oct-2018,   -2.5,    0.45, -3.100E+02,  3.100E-03,  3.100E-01,  3.100E-02,  4.100E-01,  22.50,  3.100E-01,  4.100E-01,  5.100E-01,   -2.0,   -3.0, 297,    0
24-Oct-2018,   -3.5,    0.55, -4.100E+02,  4.100E-03,  4.100E-01,  4.100E-02,  5.100E-01,  23.50,  4.100E-01,  5.100E-01,  6.100E-01,   -3.0,   -4.0, 297,    0
24-Oct-2018,   -4.5,    0.65, -5.100E+02,  5.100E-03,  5.100E-01,  5.100E-02,  6.100E-01,  24.50,  5.100E-01,  6.100E-01,  7.100E-01,   -4.0,   -5.0, 297,    0
24-Oct-2018,   -5.5,    0.75, -6.100E+02,  6.100E-03,  6.100E-01,  6.100E-02,  7.100E-01,  25.50,  6.100E-01,  7.100E-01,  8.100E-01,   -5.0,  -10.0, 297,    0
24-Oct-2018,  -10.0,                                                     0.0,  -10.0, 297,    0
"""

    path.write_text(content, encoding="latin-1")


def test_read_raw_excludes_boundary_rows(tmp_path: Path):
    vap_path = tmp_path / "result.vap"
    write_test_vap(vap_path)

    reader = VAPReader(vap_path)
    raw = reader.read_raw()

    assert len(raw) == 12
    assert raw["date"].nunique() == 2
    assert raw["depth"].min() == pytest.approx(-5.5)
    assert not (raw["depth"] == -10.0).any()


def test_requested_layer_validation():
    layer = RequestedLayer(top=0, bottom=5)

    assert layer.name == "0-5"

    with pytest.raises(ValueError):
        RequestedLayer(top=-1, bottom=5)

    with pytest.raises(ValueError):
        RequestedLayer(top=5, bottom=5)


def test_weighted_mean_uses_only_requested_layer(tmp_path: Path):
    vap_path = tmp_path / "result.vap"
    write_test_vap(vap_path)

    reader = VAPReader(vap_path)

    result = reader.extract(
        layers=[RequestedLayer(top=0, bottom=5)],
        columns=["wcontent"],
    )

    first_day = result.loc[result["date"] == pd.Timestamp("2018-10-23")]

    assert len(first_day) == 1
    assert first_day.iloc[0]["wcontent"] == pytest.approx(0.40)


def test_partial_compartment_overlap_is_weighted_correctly(tmp_path: Path):
    vap_path = tmp_path / "result.vap"
    write_test_vap(vap_path)

    reader = VAPReader(vap_path)

    result = reader.extract(
        layers=[RequestedLayer(top=4, bottom=7)],
        columns=["wcontent"],
    )

    first_day = result.loc[result["date"] == pd.Timestamp("2018-10-23")]

    # 4-5 cm: 0.60 over 1 cm
    # 5-7 cm: 0.70 over 2 cm
    expected = (0.60 * 1.0 + 0.70 * 2.0) / 3.0

    assert first_day.iloc[0]["wcontent"] == pytest.approx(expected)


def test_rootext_uses_sum(tmp_path: Path):
    vap_path = tmp_path / "result.vap"
    write_test_vap(vap_path)

    reader = VAPReader(vap_path)

    result = reader.extract(
        layers=[RequestedLayer(top=0, bottom=5)],
        columns=["rootext"],
    )

    first_day = result.loc[result["date"] == pd.Timestamp("2018-10-23")]

    expected = 0.01 + 0.02 + 0.03 + 0.04 + 0.05

    assert first_day.iloc[0]["rootext"] == pytest.approx(expected)


def test_flux_can_use_top_aggregation(tmp_path: Path):
    vap_path = tmp_path / "result.vap"
    write_test_vap(vap_path)

    reader = VAPReader(vap_path)

    result = reader.extract(
        layers=[RequestedLayer(top=0, bottom=5)],
        columns=[
            RequestedColumn(
                name="waterflux",
                aggregation="top",
            )
        ],
    )

    first_day = result.loc[result["date"] == pd.Timestamp("2018-10-23")]

    assert first_day.iloc[0]["waterflux"] == pytest.approx(0.20)


def test_multiple_layers_and_columns(tmp_path: Path):
    vap_path = tmp_path / "result.vap"
    write_test_vap(vap_path)

    reader = VAPReader(vap_path)

    result = reader.extract(
        layers=[
            RequestedLayer(top=0, bottom=5),
            RequestedLayer(top=5, bottom=10),
        ],
        columns=["wcontent", "phead", "temp"],
    )

    assert len(result) == 4
    assert list(result.columns) == [
        "date",
        "layer",
        "wcontent",
        "phead",
        "temp",
    ]

    assert set(result["layer"]) == {"0-5", "5-10"}


@pytest.mark.integration
def test_read_real_vap():
    model_dir = Path(__file__).resolve().parents[1] / "model"
    vap_path = model_dir / "results" / "result.vap"

    assert vap_path.exists(), f"VAP file not found: {vap_path}"

    reader = VAPReader(vap_path)
    raw = reader.read_raw()

    assert not raw.empty

    assert raw["date"].notna().all()
    assert raw["depth"].notna().all()
    assert raw["top"].notna().all()
    assert raw["bottom"].notna().all()

    assert (raw["bottom"] < raw["top"]).all()

    assert raw["date"].nunique() > 1
    assert raw["depth"].max() <= 0
    assert raw["depth"].min() < 0

    result = reader.extract(
        layers=[RequestedLayer(top=0, bottom=5)],
        columns=["wcontent"],
    )

    assert not result.empty
    assert result["date"].nunique() == raw["date"].nunique()
    assert result["wcontent"].notna().any()


def test_weighted_mean_aggregation():
    """Calculate a thickness-weighted mean across soil compartments."""

    reader = VAPReader(Path("dummy.vap"))

    daily = pd.DataFrame(
        {
            "top": [0.0, -5.0],
            "bottom": [-5.0, -10.0],
            "wcontent": [0.30, 0.20],
        }
    )

    value = reader._aggregate(
        daily=daily,
        layer=RequestedLayer(0, 10),
        column=RequestedColumn(
            name="wcontent",
            aggregation="weighted_mean",
        ),
    )

    assert value == pytest.approx(0.25)


def test_mean_aggregation():
    """Calculate an arithmetic mean across selected compartments."""

    reader = VAPReader(Path("dummy.vap"))

    daily = pd.DataFrame(
        {
            "top": [0.0, -5.0],
            "bottom": [-5.0, -10.0],
            "wcontent": [0.30, 0.20],
        }
    )

    value = reader._aggregate(
        daily=daily,
        layer=RequestedLayer(0, 10),
        column=RequestedColumn(
            name="wcontent",
            aggregation="mean",
        ),
    )

    assert value == pytest.approx(0.25)


def test_sum_aggregation():
    """Calculate the sum across selected compartments."""

    reader = VAPReader(Path("dummy.vap"))

    daily = pd.DataFrame(
        {
            "top": [0.0, -5.0],
            "bottom": [-5.0, -10.0],
            "rootext": [0.10, 0.20],
        }
    )

    value = reader._aggregate(
        daily=daily,
        layer=RequestedLayer(0, 10),
        column=RequestedColumn(
            name="rootext",
            aggregation="sum",
        ),
    )

    assert value == pytest.approx(0.30)


def test_top_aggregation():
    """Return the value from the uppermost selected compartment."""

    reader = VAPReader(Path("dummy.vap"))

    daily = pd.DataFrame(
        {
            "top": [0.0, -5.0],
            "bottom": [-5.0, -10.0],
            "phead": [-100.0, -200.0],
        }
    )

    value = reader._aggregate(
        daily=daily,
        layer=RequestedLayer(0, 10),
        column=RequestedColumn(
            name="phead",
            aggregation="top",
        ),
    )

    assert value == pytest.approx(-100.0)


def test_bottom_aggregation():
    """Return the value from the lowermost selected compartment."""

    reader = VAPReader(Path("dummy.vap"))

    daily = pd.DataFrame(
        {
            "top": [0.0, -5.0],
            "bottom": [-5.0, -10.0],
            "phead": [-100.0, -200.0],
        }
    )

    value = reader._aggregate(
        daily=daily,
        layer=RequestedLayer(0, 10),
        column=RequestedColumn(
            name="phead",
            aggregation="bottom",
        ),
    )

    assert value == pytest.approx(-200.0)
