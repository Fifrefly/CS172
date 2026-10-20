import io
import os
import zipfile

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F
import random

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
NUM_DIGITS = 5


class ZIP_CAPTCHA(Dataset):
    def __init__(
        self,
        source,
        image_size,
        transform: transforms.Transform | None = None,
        num_images=None,
        seed=0,
    ):
        self.source = os.fspath(source)
        self.image_size = tuple(image_size)
        self.transform = transform
        self.zip_file = None
        self.zip_pid = None
        paths = sorted(self._find_images())
        if num_images is not None:
            random.Random(seed).shuffle(paths)
            paths = sorted(paths[:num_images])
        self.img_paths = paths
        self.resize = transforms.Resize(self.image_size, antialias=True)
        self.to_float_and_normalize = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def _find_images(self):
        with zipfile.ZipFile(self.source) as archive:
            names = archive.namelist()
        return [name for name in names if name.lower().endswith(IMAGE_EXTENSIONS)]

    def _read_image(self, img_path):
        if self.zip_file is None or self.zip_pid != os.getpid():
            self.zip_file = zipfile.ZipFile(self.source)
            self.zip_pid = os.getpid()
        data = self.zip_file.read(img_path)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return F.pil_to_tensor(image)  # uint8, [3, H0, W0]

    @staticmethod
    def _parse_label(filename):
        # "12345#0.png" -> tensor([1, 2, 3, 4, 5])
        stem = os.path.splitext(os.path.basename(filename))[0]  # "12345#0"
        digits = stem.split("#", maxsplit=1)[0]  # "12345"
        # 验证长度是否正确
        assert len(digits) == NUM_DIGITS, f"Expected {NUM_DIGITS} digits, got {len(digits)} in {filename}"
        # 把每个字符转换为整数并返回一个长整型张量
        return torch.tensor([int(char) for char in digits], dtype=torch.long)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = self._read_image(img_path)  # uint8, [3, H0, W0]
        label = self._parse_label(img_path)  # long,  [5]
        image = self.resize(image)  # uint8, [3, H, W]
        # 数据增强
        if self.transform is not None:
            image = self.transform(image)
        image = self.to_float_and_normalize(image)  # float32, [3, H, W],范围约 [-1, 1]

        return image, label


class CAPTCHA(Dataset):
    def __init__(
        self,
        root,
        image_size,
        transform: transforms.Transform | None = None,
        num_images=None,
        seed=0,
    ):
        self.root = os.fspath(root)
        self.image_size = tuple(image_size)
        self.transform = transform
        paths = sorted(self._find_images())
        if num_images is not None:
            random.Random(seed).shuffle(paths)
            paths = sorted(paths[:num_images])
        self.img_paths = paths
        self.resize = transforms.Resize(self.image_size, antialias=True)
        self.to_float_and_normalize = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def _find_images(self):
        names = os.listdir(self.root)
        return [name for name in names if name.lower().endswith(IMAGE_EXTENSIONS)]

    def _read_image(self, img_path):
        image = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        return F.pil_to_tensor(image)  # uint8, [3, H0, W0]

    @staticmethod
    def _parse_label(filename):
        # "12345#0.png" -> tensor([1, 2, 3, 4, 5])
        stem = os.path.splitext(os.path.basename(filename))[0]  # "12345#0"
        digits = stem.split("#", maxsplit=1)[0]  # "12345"
        # 验证长度是否正确
        assert len(digits) == NUM_DIGITS, f"Expected {NUM_DIGITS} digits, got {len(digits)} in {filename}"
        # 把每个字符转换为整数并返回一个长整型张量
        return torch.tensor([int(char) for char in digits], dtype=torch.long)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = self._read_image(img_path)  # uint8, [3, H0, W0]
        label = self._parse_label(img_path)  # long,  [5]
        image = self.resize(image)  # uint8, [3, H, W]
        # 数据增强
        if self.transform is not None:
            image = self.transform(image)
        image = self.to_float_and_normalize(image)  # float32, [3, H, W],范围约 [-1, 1]

        return image, label
