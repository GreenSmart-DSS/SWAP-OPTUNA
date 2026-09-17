import shutil
import subprocess
from pathlib import Path

from parameters import SoilParameters, validate_parameters

NORMAL_COMPLETION_MESSAGE = "Swap normal completion!"

GENERATED_MODEL_FILES = {
    "SWAP.swp",
    "swap.ok",
    "swap_swap.log",
    "heatparam.csv",
    "soilphysparam.csv",
}

GENERATED_RESULTS_DIRECTORY = "results"


class SwapModel:
    """Interface for rendering and running the SWAP model."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.template_path = self.model_dir / "SWAP_template.txt"
        self.input_path = self.model_dir / "SWAP.swp"
        self.executable_path = self.model_dir / "Swap32.exe"
        self.vap_path = self.model_dir / "results" / "result.vap"

    @classmethod
    def create_trial_workspace(
        cls,
        source_model_dir: Path,
        trial_dir: Path,
    ) -> "SwapModel":
        """Create an isolated SWAP workspace from the reference model."""

        source_model_dir = Path(source_model_dir)
        trial_dir = Path(trial_dir)

        if not source_model_dir.exists():
            raise FileNotFoundError(
                f"Source model directory was not found: " f"{source_model_dir}"
            )

        if trial_dir.exists():
            raise FileExistsError(f"Trial workspace already exists: {trial_dir}")

        trial_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        for source_path in source_model_dir.iterdir():
            if source_path.name in GENERATED_MODEL_FILES:
                continue

            if source_path.name == GENERATED_RESULTS_DIRECTORY:
                continue

            if source_path.is_file():
                shutil.copy2(
                    source_path,
                    trial_dir / source_path.name,
                )

        results_dir = trial_dir / GENERATED_RESULTS_DIRECTORY
        results_dir.mkdir()

        return cls(trial_dir)

    def render_input(self, parameters: SoilParameters) -> Path:
        """Render SWAP.swp from the template and soil parameters."""

        validate_parameters(parameters)

        text = self.template_path.read_text(encoding="utf-8")

        values = {
            "ores": parameters.ores,
            "osat": parameters.osat,
            "alfa": parameters.alfa,
            "npar": parameters.npar,
            "ksatfit": parameters.ksatfit,
            "lexp": parameters.lexp,
            "alfaw": parameters.alfaw,
            "h_enpr": parameters.h_enpr,
            "ksatexm": parameters.ksatexm,
            "bdens": parameters.bdens,
        }

        for name, value in values.items():
            text = text.replace(
                f"{{{name}}}",
                f"{value:.7f}",
            )

        self.input_path.write_text(
            text,
            encoding="utf-8",
        )

        return self.input_path

    def clean_previous_outputs(self) -> None:
        """Remove output files from the previous SWAP run."""

        files_to_remove = [
            self.model_dir / "heatparam.csv",
            self.model_dir / "soilphysparam.csv",
            self.model_dir / "swap_swap.log",
            self.model_dir / "swap.ok",
        ]

        for path in files_to_remove:
            if path.exists():
                path.unlink()

        results_dir = self.model_dir / "results"

        if results_dir.exists():
            for path in results_dir.iterdir():
                if path.is_file() and path.name != ".gitkeep":
                    path.unlink()

    def run(
        self,
        parameters: SoilParameters,
    ) -> subprocess.CompletedProcess[str]:
        """Render the SWAP input file and run SWAP."""

        self.clean_previous_outputs()
        self.render_input(parameters)

        result = subprocess.run(
            [str(self.executable_path)],
            check=False,
            cwd=self.model_dir,
            capture_output=True,
            text=True,
        )

        if NORMAL_COMPLETION_MESSAGE not in result.stdout:
            raise RuntimeError(
                "SWAP did not report normal completion.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        if not self.vap_path.exists():
            raise RuntimeError(
                f"SWAP reported normal completion, but output was not found: "
                f"{self.vap_path}"
            )

        return result
