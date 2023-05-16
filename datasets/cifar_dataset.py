# -*- coding: utf-8 -*-
import copy
import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms
from PIL import Image

from utils.randaugment import RandAugmentMC


class CIFAR10_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mean, std, labeled_indexs, augCount=1, **kwargs):
        self.c10 = datasets.CIFAR10(root, train=True, download=False)
        self.labeled_indexs = labeled_indexs
        self.augCount = augCount

        self.transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.transform_fixMatch = TransformFixMatch(mean=mean, std=std)

    def __getitem__(self, idx):
        image, target, islabeled = self.c10.data[idx], self.c10.targets[idx], 1 if idx in self.labeled_indexs else 0
        image = Image.fromarray(image)

        imgs_labeled, imgs_strong, imgs_weak = [], [], []
        for aIdx in range(self.augCount):
            img_labeled = self.transform_labeled(copy.deepcopy(image))
            imgs_labeled.append(img_labeled)
            img_weak, img_strong = self.transform_fixMatch(copy.deepcopy(image))
            imgs_strong.append(img_strong)
            imgs_weak.append(img_weak)

        meta = {"islabeled": islabeled}
        return imgs_labeled, imgs_strong, imgs_weak, target, meta

    def __len__(self):
        return len(self.c10.targets)


class CIFAR100_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mean, std, labeled_indexs, augCount=1, **kwargs):
        self.c100 = datasets.CIFAR100(root, train=True, download=False)
        self.labeled_indexs = labeled_indexs
        self.augCount = augCount

        self.transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.transform_fixMatch = TransformFixMatch(mean=mean, std=std)

    def __getitem__(self, idx):
        image, target, islabeled = self.c100.data[idx], self.c100.targets[idx], 1 if idx in self.labeled_indexs else 0
        image = Image.fromarray(image)

        imgs_labeled, imgs_strong, imgs_weak = [], [], []
        for aIdx in range(self.augCount):
            img_labeled = self.transform_labeled(copy.deepcopy(image))
            imgs_labeled.append(img_labeled)
            img_weak, img_strong = self.transform_fixMatch(copy.deepcopy(image))
            imgs_strong.append(img_strong)
            imgs_weak.append(img_weak)

        meta = {"islabeled": islabeled}
        return imgs_labeled, imgs_strong, imgs_weak, target, meta

    def __len__(self):
        return len(self.c100.targets)


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

