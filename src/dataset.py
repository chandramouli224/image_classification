import torch
import numpy as np
from PIL import Image
from PIL import ImageFile

# sometimes we will get  iages without ending bit which are considered as corrupt.
# below will take care of such images

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    """
    A general classification dataset class which can be used for all kinds of imageclassification tasks
    """

    def __init__(self, image_paths, targets, resize=None, augmentation=None) -> None:
        """
        param image_paths: list of images
        param  targets: numpy array
        param resize: tuple, e.g. (256, 256), resizes the image if not None
        param augmentation: albuumentation of augmentaions
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentaion = augmentation

    def __len__(self):
        """
        Retuen total number of samples in dataset
        """
        return len(self.image_paths)

    def __getitem__(self, item):
        """
        For a giveen 'item' index, returns everything er need to train a given model
        """
        # use PIL to open the image
        image = Image.open(self.image_paths[item])
        # convert imageinto RGB, we have simgle channel images
        image = image.convert("RGB")
        # grab correct targets
        targets = self.targets[item]

        # resize if needed
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        # convert image into numpy array
        image = np.array(image)

        # if we have albumentation of augmentations add the to the imag
        if self.augmentaion is not None:
            augmented = self.augmentaion(image=image)
            image = augmented["image"]

        # pytorch expects CHW instead of HWC
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # return tensors of images
        # take a look at the  types
        # for regression tasks, dtype of targets will change to torch.float
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
