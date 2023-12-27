import os

import pandas as pd
import numpy as np

import albumentations
import torch
from torch.utils.data import DataLoader

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model_alexnet, get_model_resnet

if __name__ == "__main__":
    # location of train.csv and train_png folder
    # with all png files
    data_path = "C:\\Users\\mouli\\Desktop\\ML projects\\siim_dataset\\"
    # device cuda/cpu
    device = "cpu"

    # lets train for 10 epochs
    epochs = 10

    # load the  dataframe
    df = pd.read_csv(os.path.join(data_path, "train.csv"))

    # featc all image ids
    images = df.ImageId.values.tolist()

    # a list with image locations
    images = [os.path.join(data_path, "train_png", i + ".png") for i in images]

    # binary targets numpy array
    targets = df.target.values

    # fetch alexnet model, we will try both pretrained and non-pretrained weights
    # we tried with alexnet and the AUC is around 0.68 which is ok but not up to the mark
    # model = get_model_alexnet(pretrained=True)

    # lets fetch resnet18 model
    # AUC is 0.8613 which is pretty good with CPU and AUC is 0.8589 with cuda core.
    model = get_model_resnet(pretrained=True)

    # move model to device
    model.to(device)

    # mean and std values of RGB channels for imagenet dataset we use these pre-calculatted values when we use weights from imagenet
    # Remember when we do not use imagenet weights, we use the mean and std values of the original dataset.
    # Please not that these are seperate calculations

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # albumentations is an image augmentation library that allows us to  do many different types of image augmentations.
    # Here I am using normalisation and 'always_apply = true' as we want to apply noralisation
    aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )

    # instead of using kfold, I am using train_test_split from aklearn library
    train_images, valid_images, train_targets, valid_targets = train_test_split(
        images, targets, stratify=targets, random_state=42
    )

    # fetch the classificationDataset class
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=(224, 224),
        augmentation=aug,
    )

    # torch dataloader crates batches of data
    # from  classification dataset class
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # same  for validation dataset
    valid_dataset = dataset.ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(224, 224),
        augmentation=aug,
    )

    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=4)

    # sample adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # train and  print auc score for all epochs
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(valid_loader, model, device=device)
        roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch={epoch}, Valid ROC AUC = {roc_auc}")
