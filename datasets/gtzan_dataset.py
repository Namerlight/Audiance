import os
import torch
import datasets.preprocessing as preprocessing
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


def gtzan_dataset(path: str, split: str) -> torch.utils.data.Dataset:
    folder_path = os.path.join(path, split, "image")
    return ImageFolder(root=folder_path, transform=preprocessing.image_transforms("gtzan_spectrograms"))

