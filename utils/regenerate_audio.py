import os
from PIL import Image

class AudioRegenerator:
    def __init__(self, img_path: str) -> None:
        self.file_name = img_path.split(os.sep)[-1]
        self.img_data = Image.open(img_path)

    def regen_audio(self):
        """

        Returns:

        """


