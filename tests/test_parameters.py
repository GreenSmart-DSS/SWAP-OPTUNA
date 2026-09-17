import pytest

from parameters import (  # type: ignore
    SoilParameters,
    validate_parameters,
)


def test_valid_parameters():
    parameters = SoilParameters(
        ores=0.0875658534,
        osat=0.3624172551,
        alfa=0.0060393837,
        npar=1.1065721142,
        ksatfit=19.6334210991,
        lexp=0.5659299996,
        ksatexm=27.4823857627,
        bdens=1428.2709486373,
    )

    validate_parameters(parameters)

    assert parameters.alfaw == parameters.alfa
    assert parameters.h_enpr == 0.0


def test_parameter_outside_bounds():
    parameters = SoilParameters(
        ores=0.10,
        osat=0.36,
        alfa=0.006,
        npar=1.2,
        ksatfit=15.0,
        lexp=1.0,
        ksatexm=30.0,
        bdens=1450.0,
    )

    with pytest.raises(ValueError):
        validate_parameters(parameters)
