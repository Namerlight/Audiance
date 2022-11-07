import os
from PIL import Image


def crop_spectrograms(img_path: str) -> None:
    """
    Replaces image in the path with a cropped version. This is in-place, so be careful when running this function.

    Args:
        img_path: path to image to crop
    """

    img = Image.open(img_path)
    width, height = img.size

    cr_rat = {
        "tlx": 0.0610,
        "tly": 0.0346,
        "brx": 0.8771,
        "bry": 0.9505,
    }
    cropped = img.crop((int(cr_rat["tlx"] * width), int(cr_rat["tly"] * height),
                        int(cr_rat["brx"] * width), int(cr_rat["bry"] * height))
                       )
    cropped.save(img_path)