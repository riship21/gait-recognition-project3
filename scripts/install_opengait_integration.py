
"""Install Project 3 sampling support into a local OpenGait checkout.

Run from the Project 3 root:
    python -m scripts.install_opengait_integration

Does not modify Project 2 or overwrite an unknown OpenGait version.
"""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OPENGAIT = ROOT / "third_party" / "OpenGait"
TARGET = OPENGAIT / "opengait" / "data" / "collate_fn.py"

EXPECTED_COMMIT = "4d228e154ae6f7473db49e0ef33493be0c137601"

INIT_ANCHOR = (
    "        self.points_in_use = sample_config.get('points_in_use')"
)

INIT_INSERT = """        # Project 3: Optional custom inference frame selection.
        self.project3_method = sample_config.get('project3_method')
        self.project3_quality_weight = sample_config.get(
            'project3_quality_weight', 0.5
        )

        if self.project3_method is not None:
            if self.sampler != 'all' or not self.ordered:
                raise ValueError(
                    'Project 3 requires sample_type: all_ordered'
                )
            if self.frames_all_limit <= 0:
                raise ValueError(
                    'Project 3 requires a positive frames_all_limit'
                )

"""

FRAME_ANCHOR = "            for i in range(feature_num):"

FRAME_INSERT = """            # Project 3: Replace default inference indices when enabled.
            if self.project3_method is not None:
                import sys
                from pathlib import Path

                project_root = Path(__file__).resolve().parents[4]
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))

                from src.sampling.opengait_adapter import (
                    select_opengait_indices,
                )

                indices = select_opengait_indices(
                    seqs,
                    budget=self.frames_all_limit,
                    method=self.project3_method,
                    quality_weight=self.project3_quality_weight,
                )

"""


def main():
    if not TARGET.is_file():
        raise FileNotFoundError(f"Missing OpenGait file: {TARGET}")

    commit = subprocess.check_output(
        ["git", "-C", str(OPENGAIT), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    if commit != EXPECTED_COMMIT:
        raise RuntimeError("Unexpected OpenGait revision.")

    source = TARGET.read_text(encoding="utf-8")

    if "self.project3_method = sample_config.get(" in source:
        print("Project 3 integration is already installed.")
        return

    if source.count(INIT_ANCHOR) != 1:
        raise RuntimeError("Initialization anchor mismatch.")

    if source.count(FRAME_ANCHOR) != 1:
        raise RuntimeError("Frame-selection anchor mismatch.")

    # Refuse to modify an already changed OpenGait checkout.
    changes = subprocess.check_output(
        [
            "git", "-C", str(OPENGAIT),
            "status", "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()

    if changes:
        raise RuntimeError(
            "OpenGait has tracked modifications. Inspect them first."
        )

    updated = source.replace(
        INIT_ANCHOR, INIT_INSERT + INIT_ANCHOR, 1
    )

    updated = updated.replace(
        FRAME_ANCHOR, FRAME_INSERT + FRAME_ANCHOR, 1
    )

    compile(updated, str(TARGET), "exec")
    TARGET.write_text(updated, encoding="utf-8")

    print("PASS: Project 3 integration installed.")
    print(f"Modified: {TARGET}")


if __name__ == "__main__":
    main()
