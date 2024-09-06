from typing import List
from pathlib import Path

import numpy as np
from PIL import Image

from failgen.utils.image_manipulation import create_image_pack


def test_create_image_pack() -> None:
    # read in the test data ---------------------
    current_dir = Path(__file__).parent.resolve()
    test_data_dir = (current_dir.parent / "resources" / "test_data").resolve()

    front_imgs_fpaths = [
        (test_data_dir / "front" / f"{i}.png").resolve() for i in range(5)
    ]
    side_imgs_fpaths = [
        (test_data_dir / "side" / f"{i}.png").resolve() for i in range(5)
    ]
    wrist_imgs_fpaths = [
        (test_data_dir / "wrist" / f"{i}.png").resolve() for i in range(5)
    ]

    front_imgs = [np.array(Image.open(fpath)) for fpath in front_imgs_fpaths]
    side_imgs = [np.array(Image.open(fpath)) for fpath in side_imgs_fpaths]
    wrist_imgs = [np.array(Image.open(fpath)) for fpath in wrist_imgs_fpaths]
    # -------------------------------------------

    res_image = create_image_pack(
        front_view_images=front_imgs,
        side_view_images=side_imgs,
        wrist_view_images=wrist_imgs,
        start_idx=0,
        end_idx=5,
    )

    res_image.save("image_pack_result.png")

def test_create_full_pack() -> None:
    # read in the test data ---------------------
    current_dir = Path(__file__).parent.resolve()
    test_data_dir = (current_dir.parent / "resources" / "test_data").resolve()

    front_imgs_fpaths = [
        (test_data_dir / "front" / f"{i}.png").resolve() for i in range(69)
    ]
    side_imgs_fpaths = [
        (test_data_dir / "side" / f"{i}.png").resolve() for i in range(69)
    ]
    wrist_imgs_fpaths = [
        (test_data_dir / "wrist" / f"{i}.png").resolve() for i in range(69)
    ]

    front_imgs = [np.array(Image.open(fpath)) for fpath in front_imgs_fpaths]
    side_imgs = [np.array(Image.open(fpath)) for fpath in side_imgs_fpaths]
    wrist_imgs = [np.array(Image.open(fpath)) for fpath in wrist_imgs_fpaths]
    # -------------------------------------------

    res_image = create_image_pack(
        front_view_images=front_imgs,
        side_view_images=side_imgs,
        wrist_view_images=wrist_imgs,
        start_idx=0,
        end_idx=69,
    )

    res_image.save("image_pack_full_result.png")



if __name__ == "__main__":
    # test_create_image_pack()
    test_create_full_pack()
