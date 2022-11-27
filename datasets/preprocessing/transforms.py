from typing import List
from torchvision import transforms

# A dict listing all the transforms for datasets we might want to use.
transforms_dict = {
    "gtzan_spectrograms":  [
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
    ],
}

def image_transforms(dataset_name: str) -> transforms.Compose:
    """
    Get the corresponding image transformations for the particular dataset.

    Args:
        dataset_name (str): name of dataset, assuming we're using different sets of transforms for different datasets.

    Returns:
        transforms.Compose: torchvision transforms class object (I guess)
    """
    if dataset_name not in transforms_dict.keys():
        raise Exception("Dataset not in list.")

    return transforms.Compose(transforms_dict[dataset_name])


def get_list_of_transforms() -> List[str]:
    """
    Lists all datasets and transformations implemented.

    Returns:
        List[str]: list of transforms names that can be passed to image_transforms
    """
    return list(transforms_dict.keys())