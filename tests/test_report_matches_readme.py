"""The numbers in the README must be the numbers the stored verdicts give, not typed by hand."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["base", "AB2", "A3", "AB3", "AB4"]


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def test_report_rebuilds_the_committed_results_and_the_readme_table(tmp_path):
    work = tmp_path / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    subprocess.run([sys.executable, "scripts/07_report.py", *MODELS], cwd=work, check=True,
                   capture_output=True, text=True, encoding="utf-8")

    rebuilt = _lines(work / "runs" / "RESULTS.md")
    assert rebuilt == _lines(ROOT / "runs" / "RESULTS.md")

    readme_rows = [line for line in _lines(ROOT / "README.md")
                   if line.startswith(("| Book A (The Chrono Engine) /", "| Book B (The Unseen Guest) /",
                                       "| General knowledge probe", "| Everyday questions"))]
    assert len(readme_rows) == 20
    missing = [row for row in readme_rows if row not in rebuilt]
    assert not missing, f"README rows not produced by the report: {missing[:2]}"
