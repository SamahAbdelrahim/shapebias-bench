# ALICE Stimuli Guide

This folder contains two stimulus modes and two organization styles.

## Modes

- `stimuli_B_controlled_simple`
- `stimuli_A_auto_contrast`

## Non-packaged folders

Each mode has:

- `<mode>/<stl_id>/reference.png`
- `<mode>/<stl_id>/shape_match.png`
- `<mode>/manifest.csv` with columns:
  - `mode,stl_id,reference,shape_match`

## Packaged folders (benchmark-ready)

Each mode has:

- `stimuli_per_stl_packages/<mode>/<stl_id>/example_image.png`
- `stimuli_per_stl_packages/<mode>/<stl_id>/reference.png`
- `stimuli_per_stl_packages/<mode>/<stl_id>/shape_match.png`
- `stimuli_per_stl_packages/<mode>/<stl_id>/texture_match.png`
- `stimuli_per_stl_packages/<mode>/manifest.csv` with columns:
  - `mode,stl_id,example_image,reference,shape_match,texture_match`

Naming semantics:

- `example_image`: original image reference
- `reference`: same-shape rendered object
- `shape_match`: same-shape with different material
- `texture_match`: different-shape with same material as `reference`

## Repro scripts

From repo root:

```bash
# Generate non-packaged 2-image stimuli for B
ALICE_STIMULUS_MODE=B_controlled_simple bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py

# Generate non-packaged 2-image stimuli for A
ALICE_STIMULUS_MODE=A_auto_contrast bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py

# Add packaged texture_match (third test image)
bash ./run_blender.sh -b -P add_test_object_3_different_shape.py

# Normalize names/manifests after generation
python3 scripts/standardize_stimuli_naming.py
```
