from __future__ import annotations
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

def main(argv: list[str] | None = None) -> int:
    _ensure_repo_on_path()
    from search_engine.prepare_data import main as prepare_main
    return int(prepare_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
