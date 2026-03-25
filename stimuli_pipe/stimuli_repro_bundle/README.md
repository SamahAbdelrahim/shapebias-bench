# Stimuli Repro Bundle

This folder contains the files needed to reproduce the ALICE stimuli generation pipeline in another repository (for example, `shapebias-bench/stimuli_pipe`).

## Included

- `fixed_blender_centering_alice_texture.py`
- `run_blender.sh`
- `scripts/stl_spin_render.py`
- `scripts/stl_material_overlay_render.py`
- `colab_render.ipynb`
- `colab_render_drive.ipynb`
- `manifests/stimuli_B_manifest.csv`
- `manifests/stimuli_A_manifest.csv`
- `manifests/packages_B_manifest.csv`
- `manifests/packages_A_manifest.csv`

## Expected data layout

These scripts expect ALICE data at:

- `data/ALICE_stl_(Xu & Sandhofer, 2024)/stl/`
- `data/ALICE_stl_(Xu & Sandhofer, 2024)/images/`

and write outputs under:

- `data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_B_controlled_simple/`
- `data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_A_auto_contrast/`
- `data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_per_stl_packages/`

## How to use in shapebias-bench

1. Copy this folder into `shapebias-bench/stimuli_pipe/`.
2. Ensure Blender and ffmpeg are installed in the target environment.
3. From the copied bundle folder, run one mode at a time:

```bash
bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py -- \
  --dataset "data/ALICE_stl_(Xu & Sandhofer, 2024)/stl"
```

With environment variables:

```bash
ALICE_STIMULUS_MODE=B_controlled_simple \
ALICE_STIMULUS_RES=1024 \
ALICE_STIMULUS_SAMPLES=128 \
ALICE_STIMULUS_MATCH_REFERENCE=1 \
ALICE_STIMULUS_MATCH_STEP_DEG=2 \
bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py -- \
  --dataset "data/ALICE_stl_(Xu & Sandhofer, 2024)/stl"
```

Then for A:

```bash
ALICE_STIMULUS_MODE=A_auto_contrast \
ALICE_STIMULUS_RES=1024 \
ALICE_STIMULUS_SAMPLES=128 \
ALICE_STIMULUS_MATCH_REFERENCE=1 \
ALICE_STIMULUS_MATCH_STEP_DEG=2 \
bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py -- \
  --dataset "data/ALICE_stl_(Xu & Sandhofer, 2024)/stl"
```

## Notes

- The package layout and manifests are deterministic across runs (same input -> same naming).
- The generated files per shape package are:
  - `reference_image.png`
  - `test_object_1.png`
  - `test_object_2.png`
