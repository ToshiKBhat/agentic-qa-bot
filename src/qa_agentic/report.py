from pathlib import Path

def write_report(md: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qa_report.md"
    # ✅ force UTF-8 so emojis are fine on Windows too
    path.write_text(md, encoding="utf-8")
    return path