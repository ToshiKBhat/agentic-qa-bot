from pathlib import Path

def write_report(md: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qa_report.md"
    path.write_text(md)
    return path