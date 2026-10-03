"""Build the preprint PDF and a portable LaTeX source archive."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper"
DESTINATION = ROOT / "docs" / "assets" / "paper"


def main():
    if not shutil.which("tectonic"):
        raise SystemExit("Install Tectonic to build the preprint.")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="acro-paper-") as temporary:
        subprocess.run(["tectonic", "--outdir", temporary, "root.tex"], cwd=SOURCE, check=True)
        shutil.copyfile(Path(temporary) / "root.pdf", DESTINATION / "acro-preprint.pdf")
    with zipfile.ZipFile(DESTINATION / "acro-preprint-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and path.suffix not in {".aux", ".log", ".out", ".blg"} and path.name != "root.pdf":
                archive.write(path, Path("acro-preprint") / path.relative_to(SOURCE))
    print("Built docs/assets/paper/acro-preprint.pdf and acro-preprint-source.zip")


if __name__ == "__main__":
    main()
