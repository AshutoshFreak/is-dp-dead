import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PathMNIST(Dataset):
    def __init__(self, split="train", transforms=None, target_transform=None):
        """dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        """

        npz_file = np.load("../data/pathmnist.npz")

        self.split = split
        self.transforms = transforms
        self.transform_index = 0
        self.target_transform = target_transform

        X_train = npz_file["train_images"]
        Y_train = npz_file["train_labels"]
        X_val = npz_file["val_images"]
        Y_val = npz_file["val_labels"]
        X_test = npz_file["test_images"]
        Y_test = npz_file["test_labels"]

        if self.split == "train":  # 89996 images
            self.img = X_train
            self.label = Y_train
        elif self.split == "val":  # 10004  images
            self.img = X_val
            self.label = Y_val
        elif self.split == "test":  # 7180 images
            self.img = X_test
            self.label = Y_test

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)[0]
        img = Image.fromarray(np.uint8(img))

        if self.transforms is not None:
            img = self.transforms[self.transform_index](img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]
