from pathlib import Path

import pytest

from parameters import SoilParameters  # type: ignore
from swap import SwapModel  # type: ignore

LEGACY_PARAMETERS = SoilParameters(
    ores=0.0875658533777091,
    osat=0.362417255147982,
    alfa=0.00603938365321607,
    npar=1.10657211418239,
    ksatfit=19.6334210990903,
    lexp=0.565929999579225,
    ksatexm=27.4823857626600,
    bdens=1428.27094863731,
)


def test_render_input(tmp_path: Path):
    template = """\
ORES OSAT ALFA NPAR KSATFIT LEXP ALFAW H_ENPR KSATEXM BDENS
{ores} {osat} {alfa} {npar} {ksatfit} {lexp} {alfaw} {h_enpr} {ksatexm} {bdens}
"""

    model_dir = tmp_path
    template_path = model_dir / "SWAP_template.txt"
    template_path.write_text(template, encoding="utf-8")

    model = SwapModel(model_dir)

    output_path = model.render_input(LEGACY_PARAMETERS)

    assert output_path == model_dir / "SWAP.swp"
    assert output_path.exists()

    rendered = output_path.read_text(encoding="utf-8")

    assert "{ores}" not in rendered
    assert "{osat}" not in rendered
    assert "{alfa}" not in rendered
    assert "0.0875659" in rendered
    assert "0.3624173" in rendered
    assert "0.0060394" in rendered
    assert "1.1065721" in rendered
    assert "0.0000000" in rendered


@pytest.mark.integration
def test_run_real_swap():
    model_dir = Path(__file__).resolve().parents[1] / "model"
    model = SwapModel(model_dir)

    result = model.run(LEGACY_PARAMETERS)

    assert "Swap normal completion!" in result.stdout
    assert model.vap_path.exists()
    assert model.vap_path.stat().st_size > 0


def test_create_trial_workspace(tmp_path):
    source_dir = tmp_path / "model"
    source_dir.mkdir()

    static_files = [
        "Swap32.exe",
        "SWAP_template.txt",
        "CROP.crp",
        "weather.018",
        "weather.019",
    ]

    generated_files = [
        "SWAP.swp",
        "swap.ok",
        "swap_swap.log",
        "heatparam.csv",
        "soilphysparam.csv",
    ]

    for filename in static_files + generated_files:
        (source_dir / filename).write_text(
            filename,
            encoding="utf-8",
        )

    results_dir = source_dir / "results"
    results_dir.mkdir()

    (results_dir / "result.vap").write_text(
        "generated result",
        encoding="utf-8",
    )

    trial_dir = tmp_path / "trial_0000"

    model = SwapModel.create_trial_workspace(
        source_model_dir=source_dir,
        trial_dir=trial_dir,
    )

    assert model.model_dir == trial_dir

    for filename in static_files:
        assert (trial_dir / filename).exists()

    for filename in generated_files:
        assert not (trial_dir / filename).exists()

    assert (trial_dir / "results").exists()
    assert not (trial_dir / "results" / "result.vap").exists()


@pytest.mark.integration
def test_real_swap_runs_in_isolated_workspace(tmp_path):
    source_model_dir = Path("model")
    trial_dir = tmp_path / "trial_0000"

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

    source_generated_files = [
        source_model_dir / "swap.ok",
        source_model_dir / "swap_swap.log",
        source_model_dir / "heatparam.csv",
        source_model_dir / "soilphysparam.csv",
        source_model_dir / "SWAP.swp",
        source_model_dir / "results" / "result.vap",
    ]

    source_state = {
        path: path.read_bytes() if path.exists() else None
        for path in source_generated_files
    }

    model = SwapModel.create_trial_workspace(
        source_model_dir=source_model_dir,
        trial_dir=trial_dir,
    )

    result = model.run(parameters)

    assert "Swap normal completion!" in result.stdout
    assert model.vap_path.exists()

    assert (trial_dir / "SWAP.swp").exists()
    assert (trial_dir / "swap.ok").exists()
    assert (trial_dir / "swap_swap.log").exists()
    assert (trial_dir / "results" / "result.vap").exists()

    for path, original_content in source_state.items():
        if original_content is None:
            assert not path.exists()
        else:
            assert path.exists()
            assert path.read_bytes() == original_content
