"""Run without model weights: python -m unittest discover -s tests"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from ui import _device_and_dtype, editor_to_images
from input_utils import prepare_inputs


class DrawingInputsTest(unittest.TestCase):
    def setUp(self):
        self.background = Image.new("RGBA", (197, 123), (0, 255, 0, 255))
        self.layer = Image.new("RGBA", (197, 123), (0, 0, 0, 0))
        draw = ImageDraw.Draw(self.layer)
        draw.line([(15, 10), (90, 95), (177, 20)], fill=(255, 0, 0, 240), width=8)
        self.editor = {
            "background": self.background,
            "layers": [self.layer],
            "composite": Image.alpha_composite(self.background, self.layer),
        }

    def test_separate_mask_from_clean_background(self):
        image, mask = editor_to_images(self.editor)
        self.assertEqual(image.getpixel((20, 15)), (0, 255, 0))
        self.assertEqual(mask.getpixel((20, 15)), 240)
        self.assertEqual(mask.getpixel((0, 0)), 0)
        self.assertEqual(mask.size, image.size)

    def test_resize_matches_model_grid_and_original_size(self):
        image, mask = editor_to_images(self.editor)
        small, stroke, tensor, size = prepare_inputs(image, mask, 128)
        self.assertEqual(size, (197, 123))
        self.assertEqual(small.size, stroke.size)
        self.assertEqual(tensor.shape, (1, 1, small.height, small.width))
        self.assertTrue(np.asarray(stroke).any())
        self.assertEqual(small.width % 16, 0)
        self.assertEqual(small.height % 16, 0)

    def test_missing_stroke_rejected(self):
        self.editor["layers"] = []
        with self.assertRaisesRegex(ValueError, "Draw a red scribble"):
            editor_to_images(self.editor)


class DeviceSelectionTest(unittest.TestCase):
    def test_indexed_cuda_device_is_preserved(self):
        device, _ = _device_and_dtype("cuda:1", "auto")
        self.assertEqual(device, "cuda:1")


if __name__ == "__main__":
    unittest.main()
