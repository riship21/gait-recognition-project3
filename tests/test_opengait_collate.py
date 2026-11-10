"""CPU regression tests for Project 3's patched OpenGait CollateFn.

Run from the Project 3 root:
    python -m unittest discover -s tests -p test_opengait_collate.py -v

Requires NumPy, Git, the pinned OpenGait checkout, and the integration
installer to have been run. Only OpenGait's logging import is stubbed;
its frame-selection and batching code execute unchanged in these tests.
"""

import os
from pathlib import Path
import random
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.sampling.selector import METHODS, select_frames


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT = ROOT / "third_party" / "OpenGait"
TARGET = OPENGAIT / "opengait" / "data" / "collate_fn.py"
PIN = "4d228e154ae6f7473db49e0ef33493be0c137601"


class _SilentLogger:
    def log_debug(self, *args, **kwargs):
        pass


def _load_collate(source: str, name: str):
    """Load real collate code without importing OpenGait's GPU utilities."""
    module = types.ModuleType(name)
    module.__file__ = str(TARGET)

    utils_stub = types.ModuleType("utils")
    utils_stub.get_msg_mgr = lambda: _SilentLogger()

    with patch.dict(sys.modules, {"utils": utils_stub}):
        exec(
            compile(source, str(TARGET), "exec"),
            module.__dict__,
        )

    return module.CollateFn


def _sample(
    frames,
    subject="075",
    sequence="nm-05",
    view="090",
):
    """OpenGait item: feature streams and sequence metadata."""
    return ([frames], [subject, sequence, view, []])


def _config(method, budget=5, weight=0.5):
    return {
        "sample_type": "all_ordered",
        "frames_all_limit": budget,
        "project3_method": method,
        "project3_quality_weight": weight,
    }


class TestOpenGaitCollate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not TARGET.is_file():
            raise RuntimeError(
                f"Pinned OpenGait checkout is missing: {TARGET}"
            )

        current = TARGET.read_text(encoding="utf-8")

        if "self.project3_method = sample_config.get(" not in current:
            raise RuntimeError(
                "Run python -m scripts.install_opengait_integration first."
            )

        # Read the original committed source without resetting
        # or overwriting the modified OpenGait checkout.
        original = subprocess.check_output(
            [
                "git",
                "-C",
                str(OPENGAIT),
                "show",
                f"{PIN}:opengait/data/collate_fn.py",
            ],
            encoding="utf-8",
        )

        cls.Original = _load_collate(
            original,
            "_original_collate",
        )

        cls.Patched = _load_collate(
            current,
            "_project3_collate",
        )

    def setUp(self):
        rng = np.random.default_rng(42)

        self.frames = (
            rng.integers(
                0,
                2,
                size=(20, 8, 8),
                dtype=np.uint8,
            )
            * 255
        )

        self.batch = [_sample(self.frames)]
        self.labels = ["075", "076"]

    def assert_nested_equal(self, actual, expected):
        """Compare OpenGait outputs containing nested lists and arrays."""
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, expected.dtype)

        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))

            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)

        else:
            self.assertEqual(actual, expected)

    def test_disabled_all_ordered_matches_original(self):
        config = {"sample_type": "all_ordered"}

        actual = self.Patched(
            self.labels,
            config,
        )(self.batch)

        expected = self.Original(
            self.labels,
            config,
        )(self.batch)

        self.assert_nested_equal(actual, expected)

        np.testing.assert_array_equal(
            actual[0][0][0],
            self.frames,
        )

    def test_disabled_frame_cap_matches_original(self):
        config = {
            "sample_type": "all_ordered",
            "frames_all_limit": 5,
        }

        actual = self.Patched(
            self.labels,
            config,
        )(self.batch)

        expected = self.Original(
            self.labels,
            config,
        )(self.batch)

        self.assert_nested_equal(actual, expected)

        np.testing.assert_array_equal(
            actual[0][0][0],
            self.frames[:5],
        )

    def test_disabled_training_sampling_matches_original(self):
        config = {
            "sample_type": "fixed_ordered",
            "frames_num_fixed": 5,
            "frames_skip_num": 2,
        }

        # Compare identical random draws while preserving
        # the random-number-generator states for other tests.
        py_state = random.getstate()
        np_state = np.random.get_state()

        try:
            random.seed(123)
            np.random.seed(123)

            expected = self.Original(
                self.labels,
                config,
            )(self.batch)

            random.seed(123)
            np.random.seed(123)

            actual = self.Patched(
                self.labels,
                config,
            )(self.batch)

        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

        self.assert_nested_equal(actual, expected)

    def test_all_six_methods_preserve_selected_frames(self):
        self.assertEqual(
            set(METHODS),
            {
                "first_n",
                "uniform",
                "diversity",
                "quality",
                "combined",
                "adaptive",
            },
        )

        for method in METHODS:
            for budget in (1, 5, 30):
                with self.subTest(method=method, budget=budget):
                    expected, _ = select_frames(
                        self.frames,
                        budget,
                        method,
                    )

                    actual = self.Patched(
                        self.labels,
                        _config(method, budget),
                    )(self.batch)

                    np.testing.assert_array_equal(
                        actual[0][0][0],
                        expected,
                    )

                    self.assertEqual(
                        actual[0][0][0].dtype,
                        self.frames.dtype,
                    )

                    np.testing.assert_array_equal(
                        actual[4],
                        [[min(len(self.frames), budget)]],
                    )

    def test_uniform_is_not_replaced_by_first_n(self):
        actual = self.Patched(
            self.labels,
            _config("uniform"),
        )(self.batch)

        np.testing.assert_array_equal(
            actual[0][0][0],
            self.frames[[0, 4, 9, 14, 19]],
        )

        self.assertFalse(
            np.array_equal(
                actual[0][0][0],
                self.frames[:5],
            )
        )

    def test_variable_lengths_metadata_and_input_preserved(self):
        short = self.frames[:3].copy()
        original = self.frames.copy()

        batch = [
            _sample(self.frames),
            _sample(
                short,
                subject="076",
                sequence="nm-06",
                view="054",
            ),
        ]

        for method in METHODS:
            with self.subTest(method=method):
                expected, _ = select_frames(
                    self.frames,
                    5,
                    method,
                )

                actual = self.Patched(
                    self.labels,
                    _config(method),
                )(batch)

                np.testing.assert_array_equal(
                    actual[0][0][0],
                    np.concatenate(
                        [expected, short],
                        axis=0,
                    ),
                )

                np.testing.assert_array_equal(
                    actual[4],
                    [[5, 3]],
                )

                self.assertEqual(actual[1], [0, 1])
                self.assertEqual(actual[2], ["nm-05", "nm-06"])
                self.assertEqual(actual[3], ["090", "054"])

                np.testing.assert_array_equal(
                    self.frames,
                    original,
                )

                np.testing.assert_array_equal(
                    short,
                    original[:3],
                )

    def test_combined_weight_is_forwarded(self):
        for weight in (0.0, 0.5, 1.0):
            with self.subTest(weight=weight):
                expected, _ = select_frames(
                    self.frames,
                    5,
                    "combined",
                    quality_weight=weight,
                )

                actual = self.Patched(
                    self.labels,
                    _config("combined", weight=weight),
                )(self.batch)

                np.testing.assert_array_equal(
                    actual[0][0][0],
                    expected,
                )

    def test_invalid_custom_configuration_is_rejected(self):
        invalid = [
            {
                "sample_type": "all_ordered",
                "project3_method": "uniform",
            },
            _config("uniform", budget=0),
            _config("uniform", budget=-1),
            {
                **_config("uniform"),
                "sample_type": "all_unordered",
            },
            {
                **_config("uniform"),
                "sample_type": "fixed_ordered",
                "frames_num_fixed": 5,
                "frames_skip_num": 0,
            },
        ]

        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.Patched(self.labels, config)

        with self.assertRaises(ValueError):
            self.Patched(
                self.labels,
                _config("unknown"),
            )(self.batch)

    def test_empty_custom_sequence_is_rejected(self):
        batch = [
            _sample(
                np.empty((0, 8, 8), dtype=np.uint8)
            )
        ]

        for method in METHODS:
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    self.Patched(
                        self.labels,
                        _config(method),
                    )(batch)

    def test_import_from_opengait_working_directory(self):
        # A new process cannot reuse this test process's
        # previously imported Project 3 modules.
        code = r'''
import logging
from pathlib import Path
import runpy
import sys
import types

import numpy as np

stub = types.ModuleType("utils")
stub.get_msg_mgr = lambda: logging.getLogger("collate-test")
sys.modules["utils"] = stub

path = Path(sys.argv[1])
CollateFn = runpy.run_path(str(path))["CollateFn"]

frames = np.arange(
    20 * 4 * 4,
    dtype=np.uint8,
).reshape(20, 4, 4)

config = {
    "sample_type": "all_ordered",
    "frames_all_limit": 5,
    "project3_method": "uniform",
}

batch = [
    ([frames], ["075", "nm-05", "090", []])
]

out = CollateFn(["075"], config)(batch)

np.testing.assert_array_equal(
    out[0][0][0],
    frames[[0, 4, 9, 14, 19]],
)

from src.sampling import opengait_adapter

expected = (
    path.resolve().parents[4]
    / "src"
    / "sampling"
    / "opengait_adapter.py"
)

if Path(opengait_adapter.__file__).resolve() != expected:
    raise RuntimeError(
        "Imported the adapter from the wrong project."
    )

print("PASS: import and selection from the OpenGait directory")
'''

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(TARGET),
            ],
            cwd=OPENGAIT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )

        self.assertIn("PASS:", result.stdout)


if __name__ == "__main__":
    unittest.main()