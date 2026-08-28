from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(path for path in args.results.rglob("*") if path.is_file())
    records = [{"path": str(path.relative_to(args.results)), "bytes": path.stat().st_size, "sha256": digest(path)} for path in files]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "python": platform.python_version(),
        "baseline_commit": "40e7e1cd39bb1aa0994827643a128b24aa1f4fe2",
        "upstream_gcbfplus_commit": "fb449907bdbf981aa10f0edfecca02663ddc8037",
        "px4_commit": "d6f12ad1c4f70ad3230afd7d86e971421e02fef4",
        "training_seeds": [1101, 1102, 1103, 1104, 1105],
        "simulation_seeds": [31000, 31099],
        "px4_seeds": [41000, 41019],
        "files": records,
    }
    (args.results / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksum_lines = [f"{record['sha256']}  {record['path']}" for record in records]
    (args.results / "MANIFEST.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"MANIFEST_RESULT=PASS files={len(records)}")


if __name__ == "__main__":
    main()

