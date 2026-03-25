"""
Add test_object_3.png per packaged stimulus:
  - different shape (from data/random_stl)
  - same material as test_object_1 (variant 1 for each mode/shape)

Outputs:
  data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_per_stl_packages/<mode>/<stem>/test_object_3.png
  and updates each mode manifest.csv with a test_object_3 column.

Optional:
  ALICE_ONLY_STEMS=1,2,15
"""

import csv
import os
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent
_SCRIPTS = _PROJECT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import stl_spin_render as scene
import stl_material_overlay_render as mats


_ALICE = _PROJECT / "data" / "ALICE_stl_(Xu & Sandhofer, 2024)"
_PACKAGES_ROOT = _ALICE / "stimuli_per_stl_packages"
_RANDOM_STL = _PROJECT / "data" / "random_stl"

_MODE_MAP = {
    "stimuli_B_controlled_simple": "B_controlled_simple",
    "stimuli_A_auto_contrast": "A_auto_contrast",
}


def _parse_positive_int(raw: str, default: int) -> int:
    try:
        value = int(raw)
    except Exception:
        return default
    return value if value > 0 else default


def _set_dark_gray_background() -> None:
    world = scene.bpy.context.scene.world
    if world is None:
        world = scene.bpy.data.worlds.new("World")
        scene.bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new(type="ShaderNodeBackground")
    out = nodes.new(type="ShaderNodeOutputWorld")
    bg.inputs[0].default_value = (0.08, 0.08, 0.08, 1.0)
    bg.inputs[1].default_value = 0.55
    links.new(bg.outputs[0], out.inputs[0])


def _set_balanced_color_management(*, exposure: float) -> None:
    view = scene.bpy.context.scene.view_settings
    try:
        view.view_transform = "Filmic"
        view.look = "None"
    except Exception:
        pass
    view.exposure = exposure
    view.gamma = 1.0


def _rebalance_lighting_soft(object_size: float) -> None:
    bpy = scene.bpy
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    distance = object_size * 4.0
    area_specs = [
        ("FrontKey", 1200.0, (0, -distance * 1.45, object_size * 0.80), (90, 0, 0), object_size * 3.2),
        ("BackFill", 950.0, (0, distance * 1.45, object_size * 0.80), (90, 0, 180), object_size * 3.2),
        ("LeftFill", 750.0, (-distance * 1.25, 0, object_size * 0.70), (90, 0, -90), object_size * 2.9),
        ("RightFill", 750.0, (distance * 1.25, 0, object_size * 0.70), (90, 0, 90), object_size * 2.9),
        ("TopSoft", 850.0, (0, 0, distance * 1.70), (180, 0, 0), object_size * 4.0),
    ]
    for name, energy, location, rotation_deg, size in area_specs:
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.shape = "SQUARE"
        light_data.size = size
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        light_obj.rotation_euler = tuple(scene.radians(v) for v in rotation_deg)


def _configure_stimulus_render_controls() -> None:
    render = scene.bpy.context.scene.render
    cycles = scene.bpy.context.scene.cycles
    stimulus_res = _parse_positive_int(os.environ.get("ALICE_STIMULUS_RES", "").strip(), 1024)
    stimulus_samples = _parse_positive_int(os.environ.get("ALICE_STIMULUS_SAMPLES", "").strip(), 128)
    render.resolution_x = stimulus_res
    render.resolution_y = stimulus_res
    cycles.samples = stimulus_samples


def _selected_stems_from_env():
    raw = os.environ.get("ALICE_ONLY_STEMS", "").strip()
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def _seed_for_alice_stem(stem: str) -> int:
    # Must match the same seed used by test_object_1 generation.
    return scene._stable_int(str(_ALICE / "stl" / f"{stem}.stl"))


def _pick_distractor(stem: str, random_paths):
    if not random_paths:
        raise RuntimeError(f"No random STL files found in {_RANDOM_STL}")

    start = 0
    try:
        start = int(stem) % len(random_paths)
    except Exception:
        pass

    for off in range(len(random_paths)):
        cand = random_paths[(start + off) % len(random_paths)]
        # Prefer a non-matching stem id when possible.
        if cand.stem != stem:
            return cand
    return random_paths[start]


def _render_one(mode_folder: str, stimulus_mode: str, stem: str, random_paths) -> bool:
    stem_dir = _PACKAGES_ROOT / mode_folder / stem
    if not stem_dir.exists():
        return False

    needed = [
        stem_dir / "reference_image.png",
        stem_dir / "test_object_1.png",
        stem_dir / "test_object_2.png",
    ]
    if not all(p.exists() for p in needed):
        print(f"WARNING: missing required files in {stem_dir}; skipping")
        return False

    distractor_stl = _pick_distractor(stem, random_paths)
    out_png = stem_dir / "test_object_3.png"
    seed = _seed_for_alice_stem(stem)

    scene.clear_scene()
    scene.bpy.ops.wm.stl_import(filepath=str(distractor_stl))
    selected = list(scene.bpy.context.selected_objects)
    if not selected:
        print(f"WARNING: failed to import distractor STL for stem {stem}: {distractor_stl}")
        return False

    obj = selected[0]
    object_size = scene.center_and_scale_object(obj, target_size=2.0)
    scene.setup_scene(obj, object_size, material_mode="flat", material_seed=seed)
    _set_dark_gray_background()
    _set_balanced_color_management(exposure=0.20)
    _rebalance_lighting_soft(object_size)
    _configure_stimulus_render_controls()

    mats.apply_material_stimulus_variant(obj, seed, stimulus_mode=stimulus_mode, variant_index=1)
    scene.render_still(str(out_png))
    print(f"Rendered {out_png} using distractor shape {distractor_stl.name}")
    return True


def _update_manifest(mode_folder: str):
    manifest = _PACKAGES_ROOT / mode_folder / "manifest.csv"
    if not manifest.exists():
        print(f"WARNING: manifest missing for {mode_folder}: {manifest}")
        return

    with manifest.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    fields = list(rows[0].keys())
    if "test_object_3" not in fields:
        fields.append("test_object_3")

    for row in rows:
        stem = str(row.get("stl_id", "")).strip()
        row["test_object_3"] = (
            f"stimuli_per_stl_packages/{mode_folder}/{stem}/test_object_3.png"
            if stem
            else ""
        )

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Updated manifest: {manifest}")


def main():
    selected_stems = _selected_stems_from_env()
    random_paths = sorted(_RANDOM_STL.glob("*.stl"), key=lambda p: p.name.lower())

    total = 0
    for mode_folder, stimulus_mode in _MODE_MAP.items():
        mode_dir = _PACKAGES_ROOT / mode_folder
        if not mode_dir.exists():
            print(f"WARNING: mode folder missing: {mode_dir}")
            continue

        stems = sorted([p.name for p in mode_dir.iterdir() if p.is_dir()], key=lambda s: int(s) if s.isdigit() else s)
        for stem in stems:
            if selected_stems is not None and stem not in selected_stems:
                continue
            if _render_one(mode_folder, stimulus_mode, stem, random_paths):
                total += 1
        _update_manifest(mode_folder)

    print(f"Done. Rendered test_object_3 for {total} package entries.")


if __name__ == "__main__":
    main()
