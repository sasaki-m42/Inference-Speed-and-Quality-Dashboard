from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
