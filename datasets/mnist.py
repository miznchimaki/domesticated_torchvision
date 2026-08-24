import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.error import URLError

import numpy as np
import torch

from ..utils import _Image_fromarray
from .utils import _flip_byte_order, check_integrity, download_and_extract_archive, extract_archive, verify_str_arg
from .vision import VisionDataset


class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("train-labels-idx1-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("t10k-images-idx3-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("t10k-labels-idx1-ubyte.gz", "dummy_api_key_redacted_for_git"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file), weights_only=True)

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = _Image_fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            errors = []
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.mirrors, errors):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("train-labels-idx1-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("t10k-images-idx3-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("t10k-labels-idx1-ubyte.gz", "dummy_api_key_redacted_for_git"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class KMNIST(MNIST):
    """`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``KMNIST/raw/train-images-idx3-ubyte``
            and  ``KMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/kmnist/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("train-labels-idx1-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("t10k-images-idx3-ubyte.gz", "dummy_api_key_redacted_for_git"),
        ("t10k-labels-idx1-ubyte.gz", "dummy_api_key_redacted_for_git"),
    ]
    classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]


class EMNIST(MNIST):
    """`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``EMNIST/raw/train-images-idx3-ubyte``
            and  ``EMNIST/raw/t10k-images-idx3-ubyte`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    md5 = "dummy_api_key_redacted_for_git"
    splits = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
    # Merged Classes assumes Same structure for both uppercase and lowercase version
    _merged_classes = {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"}
    _all_classes = set(string.digits + string.ascii_letters)
    classes_split_dict = {
        "byclass": sorted(list(_all_classes)),
        "bymerge": sorted(list(_all_classes - _merged_classes)),
        "balanced": sorted(list(_all_classes - _merged_classes)),
        "letters": ["N/A"] + list(string.ascii_lowercase),
        "digits": list(string.digits),
        "mnist": list(string.digits),
    }

    def __init__(self, root: Union[str, Path], split: str, **kwargs: Any) -> None:
        self.split = verify_str_arg(split, "split", self.splits)
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super().__init__(root, **kwargs)
        self.classes = self.classes_split_dict[self.split]

    @staticmethod
    def _training_file(split) -> str:
        return f"training_{split}.pt"

    @staticmethod
    def _test_file(split) -> str:
        return f"test_{split}.pt"

    @property
    def _file_prefix(self) -> str:
        return f"emnist-{self.split}-{'train' if self.train else 'test'}"

    @property
    def images_file(self) -> str:
        return os.path.join(self.raw_folder, f"{self._file_prefix}-images-idx3-ubyte")

    @property
    def labels_file(self) -> str:
        return os.path.join(self.raw_folder, f"{self._file_prefix}-labels-idx1-ubyte")

    def _load_data(self):
        return read_image_file(self.images_file), read_label_file(self.labels_file)

    def _check_exists(self) -> bool:
        return all(check_integrity(file) for file in (self.images_file, self.labels_file))

    def download(self) -> None:
        """Download the EMNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        download_and_extract_archive(self.url, download_root=self.raw_folder, md5=self.md5)
        gzip_folder = os.path.join(self.raw_folder, "gzip")
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith(".gz"):
                extract_archive(os.path.join(gzip_folder, gzip_file), self.raw_folder)
        shutil.rmtree(gzip_folder)


class QMNIST(MNIST):
    """`QMNIST <https://github.com/facebookresearch/qmnist>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset whose ``raw``
            subdir contains binary files of the datasets.
        what (string,optional): Can be 'train', 'test', 'test10k',
            'test50k', or 'nist' for respectively the mnist compatible
            training set, the 60k qmnist testing set, the 10k qmnist
            examples that match the mnist testing set, the 50k
            remaining qmnist testing examples, or all the nist
            digits. The default is to select 'train' or 'test'
            according to the compatibility argument 'train'.
        compat (bool,optional): A boolean that says whether the target
            for each example is class number (for compatibility with
            the MNIST dataloader) or a torch vector containing the
            full qmnist information. Default=True.
        train (bool,optional,compatibility): When argument 'what' is
            not specified, this boolean decides whether to load the
            training set or the testing set.  Default: True.
        download (bool, optional): If True, downloads the dataset from
            the internet and puts it in root directory. If dataset is
            already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that
            takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform
            that takes in the target and transforms it.
    """

    subsets = {"train": "train", "test": "test", "test10k": "test", "test50k": "test", "nist": "nist"}
    resources: dict[str, list[tuple[str, str]]] = {  # type: ignore[assignment]
        "train": [
            (
                "https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz",
                "dummy_api_key_redacted_for_git",
            ),
            (
                "https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz",
                "dummy_api_key_redacted_for_git",
            ),
        ],
        "test": [
            (
                "https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz",
                "dummy_api_key_redacted_for_git",
            ),
            (
                "https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz",
                "dummy_api_key_redacted_for_git",
            ),
        ],
        "nist": [
            (
                "https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-images-idx3-ubyte.xz",
                "dummy_api_key_redacted_for_git",
            ),
            (
                "https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-labels-idx2-int.xz",
                "dummy_api_key_redacted_for_git",
            ),
        ],
    }
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(
        self, root: Union[str, Path], what: Optional[str] = None, compat: bool = True, train: bool = True, **kwargs: Any
    ) -> None:
        if what is None:
            what = "train" if train else "test"
        self.what = verify_str_arg(what, "what", tuple(self.subsets.keys()))
        self.compat = compat
        self.data_file = what + ".pt"
        self.training_file = self.data_file
        self.test_file = self.data_file
        super().__init__(root, train, **kwargs)

    @property
    def images_file(self) -> str:
        (url, _), _ = self.resources[self.subsets[self.what]]
        return os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0])

    @property
    def labels_file(self) -> str:
        _, (url, _) = self.resources[self.subsets[self.what]]
        return os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0])

    def _check_exists(self) -> bool:
        return all(check_integrity(file) for file in (self.images_file, self.labels_file))

    def _load_data(self):
        data = read_sn3_pascalvincent_tensor(self.images_file)
        if data.dtype != torch.uint8:
            raise TypeError(f"data should be of dtype torch.uint8 instead of {data.dtype}")
        if data.ndimension() != 3:
            raise ValueError("data should have 3 dimensions instead of {data.ndimension()}")

        targets = read_sn3_pascalvincent_tensor(self.labels_file).long()
        if targets.ndimension() != 2:
            raise ValueError(f"targets should have 2 dimensions instead of {targets.ndimension()}")

        if self.what == "test10k":
            data = data[0:10000, :, :].clone()
            targets = targets[0:10000, :].clone()
        elif self.what == "test50k":
            data = data[10000:, :, :].clone()
            targets = targets[10000:, :].clone()

        return data, targets

    def download(self) -> None:
        """Download the QMNIST data if it doesn't exist already.
        Note that we only download what has been asked for (argument 'what').
        """
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        split = self.resources[self.subsets[self.what]]

        for url, md5 in split:
            download_and_extract_archive(url, self.raw_folder, md5=md5)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        # redefined to handle the compat flag
        img, target = self.data[index], self.targets[index]
        img = _Image_fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        if self.compat:
            target = int(target[0])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def extra_repr(self) -> str:
        return f"Split: {self.what}"


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()

    # parse
    if sys.byteorder == "little" or sys.platform == "aix":
        magic = get_int(data[0:4])
        nd = magic % 256
        ty = magic // 256
    else:
        nd = get_int(data[0:1])
        ty = get_int(data[1:2]) + get_int(data[2:3]) * 256 + get_int(data[3:4]) * 256 * 256

    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    if sys.byteorder == "big" and not sys.platform == "aix":
        for i in range(len(s)):
            s[i] = int.from_bytes(s[i].to_bytes(4, byteorder="little"), byteorder="big", signed=False)

    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))

    # The MNIST format uses the big endian byte order, while `torch.frombuffer` uses whatever the system uses. In case
    # that is little endian and the dtype has more than one byte, we need to flip them.
    if sys.byteorder == "little" and parsed.element_size() > 1:
        parsed = _flip_byte_order(parsed)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 1:
        raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 3:
        raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
    return x
