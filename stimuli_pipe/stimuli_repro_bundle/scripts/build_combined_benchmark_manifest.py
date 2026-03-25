"""
Build a combined benchmark-ready manifest from packaged ALICE stimuli.

Inputs:
  - data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_per_stl_packages/stimuli_A_auto_contrast/manifest.csv
  - data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_per_stl_packages/stimuli_B_controlled_simple/manifest.csv

Output:
  - data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_per_stl_packages/combined_benchmark_manifest.csv

Columns:
  trial_id,mode,stl_id,example_image,target,shape_match,texture_match,distractor
"""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ALICE = PROJECT / "data" / "ALICE_stl_(Xu & Sandhofer, 2024)"
PACKAGES = ALICE / "stimuli_per_stl_packages"

MANIFESTS = [
    PACKAGES / "stimuli_A_auto_contrast" / "manifest.csv",
    PACKAGES / "stimuli_B_controlled_simple" / "manifest.csv",
]
OUT = PACKAGES / "combined_benchmark_manifest.csv"


def _mode_tag(mode_value: str) -> str:
    if "A_auto_contrast" in mode_value:
        return "A"
    if "B_controlled_simple" in mode_value:
        return "B"
    return "U"


def main() -> None:
    rows_out = []
    for manifest in MANIFESTS:
        if not manifest.exists():
            continue
        with manifest.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mode = str(row.get("mode", "")).strip()
                stl_id = str(row.get("stl_id", "")).strip()
                if not mode or not stl_id:
                    continue
                rows_out.append(
                    {
                        "trial_id": f"{_mode_tag(mode)}_{int(stl_id):03d}",
                        "mode": mode,
                        "stl_id": stl_id,
                        "example_image": row.get("example_image", ""),
                        "target": row.get("reference", ""),
                        "shape_match": row.get("shape_match", ""),
                        "texture_match": row.get("texture_match", ""),
                        # Placeholder for future negative-control assignment.
                        "distractor": "",
                    }
                )

    rows_out.sort(key=lambda r: (r["mode"], int(r["stl_id"])))

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "mode",
                "stl_id",
                "example_image",
                "target",
                "shape_match",
                "texture_match",
                "distractor",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
