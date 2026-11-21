"""Environment checks for the Python 3.11 validator and CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
REPO_ROOT = PACKAGE_ROOT.parent

_IMPORT_SNIPPET = (
    "import member1_text; "
    "print(member1_text.__version__); "
    "print(member1_text.REQUIRED_PYTHON)"
)


def _python311_executable() -> str | None:
    override = os.environ.get("MEMBER1_PYTHON311")
    if override:
        return override
    found = shutil.which("python3.11")
    if found:
        return found
    brew = Path("/opt/homebrew/opt/python@3.11/bin/python3.11")
    if brew.is_file():
        return str(brew)
    uv_root = Path.home() / ".local/share/uv/python"
    if uv_root.is_dir():
        matches = sorted(uv_root.glob("cpython-3.11-*/bin/python3.11"))
        if matches:
            return str(matches[-1])
    return None


def _run(executable: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env.pop("PYTHONSTARTUP", None)
    return subprocess.run(
        [executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_import(executable: str) -> subprocess.CompletedProcess[str]:
    return _run(executable, ["-c", _IMPORT_SNIPPET])


def _run_cli(executable: str) -> subprocess.CompletedProcess[str]:
    return _run(executable, ["-m", "member1_text"])


class Python311GateTest(unittest.TestCase):
    def test_current_process_major_minor_is_recorded(self) -> None:
        major, minor = sys.version_info[:2]
        self.assertGreaterEqual(major, 3)
        self.assertNotEqual((major, minor), (0, 0))

    def test_import_does_not_exit(self) -> None:
        result = _run_import(sys.executable)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn("0.1.0", result.stdout)
        self.assertIn("(3, 11)", result.stdout)
        self.assertNotIn("sys.exit", (SRC / "member1_text" / "__init__.py").read_text(encoding="utf-8"))

    def test_non_311_cli_exits_with_a_clear_message(self) -> None:
        if sys.version_info[:2] == (3, 11):
            self.skipTest("This test process is already Python 3.11.")
        result = _run_cli(sys.executable)
        self.assertNotEqual(result.returncode, 0)
        message = result.stderr + result.stdout
        self.assertIn("member1_text requires Python 3.11", message)
        self.assertIn(f"Python {sys.version_info.major}.{sys.version_info.minor}", message)

    def test_python_311_imports_the_package(self) -> None:
        executable = _python311_executable()
        if executable is None:
            self.fail("Python 3.11 was not found on PATH or at the Homebrew prefix.")
        result = _run_import(executable)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn("0.1.0", result.stdout)
        self.assertIn("(3, 11)", result.stdout)

    def test_python_311_cli_prints_version(self) -> None:
        executable = _python311_executable()
        if executable is None:
            self.fail("Python 3.11 was not found on PATH or at the Homebrew prefix.")
        result = _run_cli(executable)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn("0.1.0", result.stdout)


if __name__ == "__main__":
    unittest.main()
