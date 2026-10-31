"""Download the pre-trained demo artifacts from a GitHub Release into this repo.

Usage:
    python scripts/download_models.py                 # default tag/repo
    python scripts/download_models.py --tag models-v2

Extracts models/checkpoints/ and outputs/metrics/ so that
`streamlit run app/streamlit_app.py` works without training anything.
Uses only the standard library.
"""
import argparse
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "YeeFei93/VisionInspectAI"
DEFAULT_TAG = "models-v1"
ASSET_PREFIX = "visioninspect-models-"


def list_assets(repo: str, tag: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request) as response:
        release = json.load(response)
    assets = [
        asset
        for asset in release["assets"]
        if asset["name"].startswith(ASSET_PREFIX) and asset["name"].endswith(".zip")
    ]
    if not assets:
        raise SystemExit(f"Release {tag} has no {ASSET_PREFIX}*.zip assets.")
    return assets


def download(url: str, destination: Path, expected_size: int) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        received = 0
        while chunk := response.read(1 << 20):
            output.write(chunk)
            received += len(chunk)
            print(f"\r  {received / 1e6:.0f}/{expected_size / 1e6:.0f} MB", end="", flush=True)
    print()
    if received != expected_size:
        raise SystemExit(f"Incomplete download of {destination.name}; please re-run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    try:
        assets = list_assets(args.repo, args.tag)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Could not read release '{args.tag}' of {args.repo}: {error}")

    total_mb = sum(asset["size"] for asset in assets) / 1e6
    print(f"Downloading {len(assets)} archive(s), {total_mb:.0f} MB total")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for asset in assets:
            print(asset["name"])
            zip_path = Path(tmp_dir) / asset["name"]
            download(asset["browser_download_url"], zip_path, asset["size"])
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(PROJECT_ROOT)
            zip_path.unlink()
    print("Done. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    sys.exit(main())
