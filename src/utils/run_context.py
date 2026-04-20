from datetime import datetime
from pathlib import Path

def make_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def make_run_path(root_dir: Path, run_id: str) -> Path:
    out_dir = Path(root_dir) / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"run_{run_id}.json"