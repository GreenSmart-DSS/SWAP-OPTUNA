from dataclasses import dataclass


@dataclass(frozen=True)
class SoilParameters:
    """Eight soil parameters used in the legacy MATLAB calibration."""

    ores: float
    osat: float
    alfa: float
    npar: float
    ksatfit: float
    lexp: float
    ksatexm: float
    bdens: float

    @property
    def alfaw(self) -> float:
        """Wetting-curve alpha parameter."""
        return self.alfa

    @property
    def h_enpr(self) -> float:
        """Air-entry pressure head."""
        return 0.0


PARAMETER_BOUNDS = {
    "ores": (0.06, 0.09),
    "osat": (0.25, 0.50),
    "alfa": (0.006, 0.0082),
    "npar": (1.10, 1.61),
    "ksatfit": (10.01, 20.00),
    "lexp": (0.40, 1.99),
    "ksatexm": (20.01, 40.01),
    "bdens": (1350.01, 1600.01),
}


def validate_parameters(parameters: SoilParameters) -> None:
    """Validate parameters against the legacy MATLAB bounds."""

    for name, (lower, upper) in PARAMETER_BOUNDS.items():
        value = getattr(parameters, name)

        if not lower <= value <= upper:
            raise ValueError(
                f"{name}={value} is outside the allowed range " f"[{lower}, {upper}]."
            )
