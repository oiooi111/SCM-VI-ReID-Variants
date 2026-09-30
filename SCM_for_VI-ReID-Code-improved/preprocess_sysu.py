"""Prepare SYSU-MM01 train/validation images for the existing data loader.

The output image resolution (192 x 384) follows the supplied preprocessing
script. The training transforms subsequently crop images to 144 x 288.
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


RGB_CAMERAS = ("cam1", "cam2", "cam4", "cam5")
IR_CAMERAS = ("cam3", "cam6")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUTS = (
    "train_rgb_resized_img.npy",
    "train_rgb_resized_label.npy",
    "train_ir_resized_img.npy",
    "train_ir_resized_label.npy",
)


def read_ids(path):
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Empty identity list: {path}")
    return {int(item.strip()) for item in raw.replace("\n", ",").split(",") if item.strip()}


def image_paths(root, identities, cameras):
    paths = []
    for identity in identities:
        for camera in cameras:
            folder = root / camera / f"{identity:04d}"
            if folder.is_dir():
                paths.extend(sorted(
                    path for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ))
    return paths


def write_arrays(image_paths_list, image_path, label_path, labels):
    image_array = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8,
        shape=(len(image_paths_list), 384, 192, 3))
    label_array = np.lib.format.open_memmap(
        label_path, mode="w+", dtype=np.int64, shape=(len(image_paths_list),))

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for index, path in enumerate(image_paths_list):
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((192, 384), resampling)
            image_array[index] = np.asarray(resized, dtype=np.uint8)
        label_array[index] = labels[int(path.parent.name)]
    image_array.flush()
    label_array.flush()
    del image_array, label_array


def preprocess(root, overwrite=False):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SYSU-MM01 directory not found: {root}")

    outputs = [root / name for name in OUTPUTS]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Preprocessed outputs already exist; use --overwrite to replace them: "
            + ", ".join(str(path) for path in existing))

    identities = sorted(
        read_ids(root / "exp/train_id.txt") |
        read_ids(root / "exp/val_id.txt"))
    rgb_paths = image_paths(root, identities, RGB_CAMERAS)
    ir_paths = image_paths(root, identities, IR_CAMERAS)
    rgb_ids = {int(path.parent.name) for path in rgb_paths}
    ir_ids = {int(path.parent.name) for path in ir_paths}
    missing = set(identities) - (rgb_ids & ir_ids)
    if missing:
        raise ValueError(
            "Missing RGB or IR training images for identities: "
            + ", ".join(f"{identity:04d}" for identity in sorted(missing)))
    labels = {identity: index for index, identity in enumerate(identities)}

    temporary = []
    try:
        for name in OUTPUTS:
            with tempfile.NamedTemporaryFile(
                    prefix=f".{name}.", suffix=".npy", dir=root,
                    delete=False) as handle:
                temporary.append(Path(handle.name))
        write_arrays(rgb_paths, temporary[0], temporary[1], labels)
        write_arrays(ir_paths, temporary[2], temporary[3], labels)
        for source, target in zip(temporary, outputs):
            os.replace(source, target)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)

    print(
        f"Prepared {len(identities)} identities: "
        f"{len(rgb_paths)} RGB images, {len(ir_paths)} IR images in {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", required=True,
        help="SYSU-MM01 root containing exp/train_id.txt and cam1..cam6")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing four .npy outputs")
    args = parser.parse_args()
    preprocess(args.data_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
