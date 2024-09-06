from math import ceil
from typing import List

from PIL import Image, ImageDraw, ImageFont
import numpy as np

NUM_TIMESTEPS = 5


def create_image_pack(
    front_view_images: List[np.ndarray],
    side_view_images: List[np.ndarray],
    wrist_view_images: List[np.ndarray],
    start_idx: int = 0,
    end_idx: int = 5,
    separator_width: int = 10,
    font_size: int = 40,
    use_annotations: bool = True,
) -> Image.Image:
    assert front_view_images[0].shape == side_view_images[0].shape
    assert side_view_images[0].shape == wrist_view_images[0].shape

    step_idx = ceil((end_idx - start_idx) / NUM_TIMESTEPS)

    img_width = front_view_images[0].shape[1]
    img_height = front_view_images[0].shape[0]
    # fmt: off
    total_width = NUM_TIMESTEPS * img_width + \
                  (NUM_TIMESTEPS - 1) * separator_width
    # fmt: on
    total_height = 3 * img_height

    image_pack = np.zeros((total_height, total_width, 3))

    j = 0
    for j_idx in range(start_idx, end_idx, step_idx):
        img_0j = front_view_images[j_idx]
        img_1j = side_view_images[j_idx]
        img_2j = wrist_view_images[j_idx]

        # Place the front view into corresponding section
        start_px_0j = (
            0 * img_height,
            (0 + j) * img_width + j * separator_width,
        )
        end_px_0j = (1 * img_height, (1 + j) * img_width + j * separator_width)
        image_pack[
            start_px_0j[0] : end_px_0j[0], start_px_0j[1] : end_px_0j[1], :
        ] = img_0j

        # Place the side view into corresponding section
        start_px_1j = (
            1 * img_height,
            (0 + j) * img_width + j * separator_width,
        )
        end_px_1j = (2 * img_height, (1 + j) * img_width + j * separator_width)
        image_pack[
            start_px_1j[0] : end_px_1j[0], start_px_1j[1] : end_px_1j[1], :
        ] = img_1j

        # Place the wrist view into corresponding section
        start_px_2j = (
            2 * img_height,
            (0 + j) * img_width + j * separator_width,
        )
        end_px_2j = (3 * img_height, (1 + j) * img_width + j * separator_width)
        image_pack[
            start_px_2j[0] : end_px_2j[0], start_px_2j[1] : end_px_2j[1], :
        ] = img_2j

        j += 1

    image_pack_pil = Image.fromarray(image_pack.astype(np.uint8))

    if use_annotations:
        # here come the number annotations
        draw = ImageDraw.Draw(image_pack_pil)
        try:
            font = ImageFont.load_default(size=font_size)
        except IOError:
            font = ImageFont.truetype("arial", font_size)

        for j in range(5):
            for i in range(3):
                start_px = (i * img_height, j * img_width + j * separator_width)
                text_px = (start_px[1] + 10, start_px[0] + 10)
                draw.text(text_px, str(j + 1), fill="white", font=font)

    return image_pack_pil
