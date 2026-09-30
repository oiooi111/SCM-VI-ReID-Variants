import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "SCM_for_VI-ReID-Code-improved",
    "SCM_for_VI-ReID-v1-hc",
    "SCM_for_VI-ReID-v2-circle",
    "SCM_for_VI-ReID-v3-mixstyle",
    "SCM_for_VI-ReID-v4-adamw-cosine",
    "SCM_for_VI-ReID-v5-combo",
)


class PreprocessSysuTest(unittest.TestCase):
    def test_every_variant_writes_images_labels_and_paths(self):
        """Catches wrong size, missing path arrays and inconsistent labels."""
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                dataset = Path(temp)
                (dataset / "exp").mkdir()
                (dataset / "exp/train_id.txt").write_text("10\n")
                (dataset / "exp/val_id.txt").write_text("2\n")
                for camera, identity, color in (
                    ("cam1", "0010", (255, 0, 0)),
                    ("cam2", "0002", (0, 0, 255)),
                    ("cam3", "0010", (120, 120, 120)),
                    ("cam6", "0002", (80, 80, 80)),
                ):
                    folder = dataset / camera / identity
                    folder.mkdir(parents=True)
                    Image.new("RGB", (8, 12), color).save(folder / "0001.jpg")

                command = [sys.executable, str(ROOT / variant / "preprocess_sysu.py"),
                           "--data-path", str(dataset)]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

                rgb = np.load(dataset / "train_rgb_resized_img.npy")
                ir = np.load(dataset / "train_ir_resized_img.npy")
                rgb_labels = np.load(dataset / "train_rgb_resized_label.npy")
                ir_labels = np.load(dataset / "train_ir_resized_label.npy")
                rgb_paths = np.load(dataset / "train_rgb_resized_path.npy")
                ir_paths = np.load(dataset / "train_ir_resized_path.npy")
                self.assertEqual(rgb.shape, (2, 288, 144, 3))
                self.assertEqual(ir.shape, (2, 288, 144, 3))
                np.testing.assert_array_equal(rgb_labels, [0, 1])
                np.testing.assert_array_equal(ir_labels, [0, 1])
                self.assertEqual(
                    rgb_paths.tolist(),
                    [str((dataset / "cam2/0002/0001.jpg").resolve()),
                     str((dataset / "cam1/0010/0001.jpg").resolve())])
                self.assertEqual(
                    ir_paths.tolist(),
                    [str((dataset / "cam6/0002/0001.jpg").resolve()),
                     str((dataset / "cam3/0010/0001.jpg").resolve())])
                self.assertEqual(rgb.dtype, np.uint8)

                repeat = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(repeat.returncode, 0)
                np.testing.assert_array_equal(
                    np.load(dataset / "train_rgb_resized_label.npy"), [0, 1])

                overwrite = subprocess.run(
                    command + ["--overwrite"], capture_output=True, text=True)
                self.assertEqual(overwrite.returncode, 0, overwrite.stderr)


if __name__ == "__main__":
    unittest.main()
