from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast
import unittest
from unittest.mock import patch


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "script" / "verify-model-improvement.py"
    spec = importlib.util.spec_from_file_location("verify_model_improvement", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load model-improvement verification script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_script()


class ModelImprovementCliTest(unittest.TestCase):
    def test_parser_defaults_to_python_and_forwards_native_options(self) -> None:
        build_parser = cast(
            Callable[[], argparse.ArgumentParser],
            getattr(_SCRIPT, "build_parser"),
        )
        parser = cast(argparse.ArgumentParser, build_parser())

        defaults = parser.parse_args(["candidate", "baseline"])
        native = parser.parse_args(
            [
                "candidate",
                "baseline",
                "--search-backend",
                "cpp",
                "--native-search-threads",
                "4",
            ]
        )

        self.assertEqual("python", defaults.search_backend)
        self.assertEqual(1, defaults.native_search_threads)
        self.assertEqual("cpp", native.search_backend)
        self.assertEqual(4, native.native_search_threads)

    def test_preflight_only_loads_native_extension_for_positive_cpp_search(self) -> None:
        preflight = cast(
            Callable[[str, int], None],
            getattr(_SCRIPT, "_preflight_native_search"),
        )

        with patch.object(_SCRIPT, "load_native_extension") as load:
            preflight("python", 16)
            preflight("cpp", 0)
            load.assert_not_called()

            preflight("cpp", 1)
            load.assert_called_once_with()

    def test_nonpositive_native_threads_fail_before_native_preflight(self) -> None:
        main = cast(
            Callable[[list[str] | None], int],
            getattr(_SCRIPT, "main"),
        )

        with (
            patch.object(
                _SCRIPT,
                "load_native_extension",
                side_effect=AssertionError("native preflight ran too early"),
            ),
            self.assertRaisesRegex(ValueError, "must be positive"),
        ):
            main(
                [
                    "candidate",
                    "baseline",
                    "--search-backend",
                    "cpp",
                    "--native-search-threads",
                    "0",
                ]
            )


if __name__ == "__main__":
    unittest.main()
