"""Package research evidence locally for sharing, excluding datasets."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "handoff_packages" / "20260927_diagnostics"
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Handoff already exists; use a new package name.")
    sources = [p for p in root.iterdir() if p.is_file() and
               (p.suffix in {".py", ".md", ".yaml"} or p.name == ".gitignore")]
    for folder in ("dstg", "scripts", "tests", "docs", "patches", "results", "scores"):
        sources.extend(p for p in (root / folder).rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts and
                       p.suffix in {".py", ".md", ".yaml", ".json", ".pt", ".npz", ".log", ".diff", ".txt"})
    sources.extend(root / "models" / f"V5_summe_{split}_0.pt" for split in range(5))
    # Official split definitions are small metadata, not video/feature datasets.
    sources.extend((root / "DSNet" / "splits").glob("*.yml"))
    destination.mkdir(parents=True)
    manifest = {}
    for source in sorted(set(sources)):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = {
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (destination / "START_HERE.md").write_text(
        "# Research handoff: September 27, 2026\n\n"
        "Start with docs/diagnostics.md and decisions.md.\n\n"
        "models/: five official V5 final checkpoints.\n"
        "results/validation/: fifteen training-only validation runs and checkpoints.\n"
        "results/diagnostics/: checkpoint checks, 100 random draws, and five graph-free runs.\n"
        "scores/: saved frame scores. Source files and configs are included.\n\n"
        "Datasets are excluded; use the team's existing licensed dataset installation.\n"
        "Historical source is in results/diagnostics/20260927_signal/source_before/.\n"
        "MANIFEST.json lists SHA-256 checksums of copied files.\n\n"
        "This is a local package. Upload it to the team's chosen Google Drive folder\n"
        "and grant Dr. Wang and Dakota access before sending a sharing link.\n",
        encoding="utf-8")
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(destination.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, metadata in manifest.items():
            data = z.read(f"{destination.name}/{name}")
            assert hashlib.sha256(data).hexdigest() == metadata["sha256"]
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "copied_files": len(manifest), "verified": True}))


if __name__ == "__main__":
    main()
