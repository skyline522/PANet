# """
# Customized data transforms for few‑shot semantic segmentation
# All geometric transforms are now applied **jointly** to the image and all
# mask‑like tensors (label / inst / scribble) so that they stay perfectly
# aligned.  Photometric transforms are applied **only** on the image.

# Author: (your name)
# Date: 2025‑07‑28
# """
# from __future__ import annotations

# import random
# from typing import Dict, Tuple, Sequence

# import numpy as np
# import torch
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF
# from PIL import Image

# __all__: Sequence[str] = [
#     "JointRandomResizedCrop", "JointRandomHorizontalFlip", "StrongAug",
#     "Resize", "ToTensorNormalize"
# ]

# # -----------------------------------------------------------------------------
# # Helper joint geometric transforms
# # -----------------------------------------------------------------------------
# class JointRandomResizedCrop:
#     """Apply the *same* RandomResizedCrop to image + masks."""

#     def __init__(self,
#                  size: Tuple[int, int] | int,
#                  scale: Tuple[float, float] = (0.5, 1.0),
#                  ratio: Tuple[float, float] = (0.8, 1.2)) -> None:
#         self.size = size
#         self.scale = scale
#         self.ratio = ratio

#     def __call__(self, sample: Dict) -> Dict:
#         img = sample["image"]
#         # sample crop params once and reuse
#         i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)

#         sample["image"] = TF.resized_crop(img, i, j, h, w, self.size,
#                                            interpolation=Image.BILINEAR)
#         for key in ("label", "inst", "scribble"):
#             sample[key] = TF.resized_crop(sample[key], i, j, h, w, self.size,
#                                           interpolation=Image.NEAREST)
#         return sample


# class JointRandomHorizontalFlip:
#     """Horizontal‑flip image + masks together."""

#     def __init__(self, p: float = 0.5) -> None:
#         self.p = p

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() < self.p:
#             for key in ("image", "label", "inst", "scribble"):
#                 sample[key] = TF.hflip(sample[key])
#         return sample


# # -----------------------------------------------------------------------------
# # Public transforms
# # -----------------------------------------------------------------------------
# class StrongAug:
#     """(Stronger) data augmentation used in training.

#     The pipeline is split into *geometric* part (affects masks) and
#     *photometric* part (only affects the image).
#     """

#     def __init__(self, size: Tuple[int, int] | int) -> None:
#         self.geom = T.Compose([
#             JointRandomResizedCrop(size),
#             JointRandomHorizontalFlip(p=0.5),
#         ])
#         self.photo = T.Compose([
#             T.ColorJitter(0.4, 0.4, 0.4, 0.1),
#             T.RandomGrayscale(p=0.2),
#             T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
#         ])

#     def __call__(self, sample: Dict) -> Dict:
#         sample = self.geom(sample)
#         # photometric only on image
#         sample["image"] = self.photo(sample["image"])
#         return sample


# class Resize:
#     """Deterministic resize (used in val/test)."""

#     def __init__(self, size: Tuple[int, int] | int):
#         self.size = size

#     def __call__(self, sample: Dict) -> Dict:
#         sample["image"] = TF.resize(sample["image"], self.size,
#                                      interpolation=Image.BILINEAR)
#         for key in ("label", "inst", "scribble"):
#             sample[key] = TF.resize(sample[key], self.size,
#                                     interpolation=Image.NEAREST)
#         return sample


# class ToTensorNormalize:
#     """Convert PIL images to *torch.Tensor* and normalize them."""

#     def __init__(self,
#                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
#                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample: Dict) -> Dict:
#         sample["image"] = TF.to_tensor(sample["image"])
#         sample["image"] = TF.normalize(sample["image"], self.mean, self.std)

#         # convert masks to tensors (no scaling!)
#         for key in ("label", "inst", "scribble"):
#             sample[key] = torch.from_numpy(np.array(sample[key], dtype=np.uint8)).long()
#         return sample
    

# """
# Customized data transforms for few-shot semantic segmentation
# All geometric transforms are now applied **jointly** to the image and all
# mask-like tensors (label / inst / scribble) so that they stay perfectly
# aligned.  Photometric transforms are applied **only** on the image.

# Author: (your name)
# Date: 2025-07-28
# """
# from __future__ import annotations

# import random
# from typing import Dict, Tuple, Sequence

# import numpy as np
# import torch
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF
# from PIL import Image

# __all__: Sequence[str] = [
#     "JointRandomResizedCrop", "JointRandomHorizontalFlip", "StrongAug",
#     "Resize", "ToTensorNormalize"
# ]

# # -----------------------------------------------------------------------------
# # Helper joint geometric transforms
# # -----------------------------------------------------------------------------
# class JointRandomResizedCrop:
#     """Apply the *same* RandomResizedCrop to image + masks."""

#     def __init__(self,
#                  size: Tuple[int, int] | int,
#                  scale: Tuple[float, float] = (0.5, 1.0),
#                  ratio: Tuple[float, float] = (0.8, 1.2)) -> None:
#         self.size = size
#         self.scale = scale
#         self.ratio = ratio

#     def __call__(self, sample: Dict) -> Dict:
#         img = sample["image"]
#         # sample crop params once and reuse
#         i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)

#         sample["image"] = TF.resized_crop(img, i, j, h, w, self.size,
#                                            interpolation=Image.BILINEAR)
#         for key in ("label", "inst", "scribble"):
#             sample[key] = TF.resized_crop(sample[key], i, j, h, w, self.size,
#                                           interpolation=Image.NEAREST)
        
#         # DEBUG: 添加尺寸检查 (用于调试尺寸不一致问题)
#         # print(f"DEBUG - StrongAug - After JointRandomResizedCrop:")
#         # print(f"  Image PIL size: {sample['image'].size}")
#         # print(f"  Label PIL size: {sample['label'].size}")
#         # print(f"  Inst PIL size: {sample['inst'].size}")
#         # print(f"  Scribble PIL size: {sample['scribble'].size}")
        
#         return sample


# class JointRandomHorizontalFlip:
#     """Horizontal-flip image + masks together."""

#     def __init__(self, p: float = 0.5) -> None:
#         self.p = p

#     def __call__(self, sample: Dict) -> Dict:
#         if random.random() < self.p:
#             for key in ("image", "label", "inst", "scribble"):
#                 sample[key] = TF.hflip(sample[key])
#         return sample


# # -----------------------------------------------------------------------------
# # Public transforms
# # -----------------------------------------------------------------------------
# class StrongAug:
#     """(Stronger) data augmentation used in training.

#     The pipeline is split into *geometric* part (affects masks) and
#     *photometric* part (only affects the image).
#     """

#     def __init__(self, size: Tuple[int, int] | int) -> None:
#         self.geom = T.Compose([
#             JointRandomResizedCrop(size),
#             JointRandomHorizontalFlip(p=0.5),
#         ])
#         self.photo = T.Compose([
#             T.ColorJitter(0.4, 0.4, 0.4, 0.1),
#             T.RandomGrayscale(p=0.2),
#             T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
#         ])

#     def __call__(self, sample: Dict) -> Dict:
#         sample = self.geom(sample)
#         # photometric only on image
#         sample["image"] = self.photo(sample["image"])
#         return sample


# class Resize:
#     """Deterministic resize (used in val/test)."""

#     def __init__(self, size: Tuple[int, int] | int):
#         self.size = size

#     def __call__(self, sample: Dict) -> Dict:
#         sample["image"] = TF.resize(sample["image"], self.size,
#                                      interpolation=Image.BILINEAR)
#         for key in ("label", "inst", "scribble"):
#             sample[key] = TF.resize(sample[key], self.size,
#                                     interpolation=Image.NEAREST)
#         return sample


# class ToTensorNormalize:
#     """Convert PIL images to *torch.Tensor* and normalize them."""

#     def __init__(self,
#                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
#                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample: Dict) -> Dict:
#         sample["image"] = TF.to_tensor(sample["image"])
#         sample["image"] = TF.normalize(sample["image"], self.mean, self.std)

#         # convert masks to tensors (no scaling!)
#         for key in ("label", "inst", "scribble"):
#             sample[key] = torch.from_numpy(np.array(sample[key], dtype=np.uint8)).long()
#             # DEBUG: 打印转换后的Tensor形状 (用于调试尺寸不一致问题)
#             # print(f"DEBUG - ToTensorNormalize - {key} tensor shape: {sample[key].shape}")
#         return sample
    
    # 文件名: transforms.py (最终修改版)
"""
Customized data transforms for few-shot semantic segmentation
All geometric transforms are now applied **jointly** to the image and all
mask-like tensors (label / inst / scribble) so that they stay perfectly
aligned.  Photometric transforms are applied **only** on the image.

Author: (your name)
Date: 2025-07-28
"""
from __future__ import annotations

import random
from typing import Dict, Tuple, Sequence

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

__all__: Sequence[str] = [
    "JointRandomResizedCrop", "JointRandomHorizontalFlip", "StrongAug",
    "Resize", "ToTensorNormalize"
]

# -----------------------------------------------------------------------------
# Helper joint geometric transforms
# -----------------------------------------------------------------------------
class JointRandomResizedCrop:
    """Apply the *same* RandomResizedCrop to image + masks."""

    def __init__(self,
                 size: Tuple[int, int] | int,
                 scale: Tuple[float, float] = (0.5, 1.0),
                 ratio: Tuple[float, float] = (0.8, 1.2)) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample: Dict) -> Dict:
        img = sample["image"]
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)

        sample["image"] = TF.resized_crop(img, i, j, h, w, self.size,
                                           interpolation=Image.BILINEAR)
        for key in ("label", "inst", "scribble"):
            sample[key] = TF.resized_crop(sample[key], i, j, h, w, self.size,
                                          interpolation=Image.NEAREST)
        
        # DEBUG: 打印尺寸信息
        # print(f"DEBUG - StrongAug - After JointRandomResizedCrop:")
        # print(f"  Image PIL size: {sample['image'].size}")
        # print(f"  Label PIL size: {sample['label'].size}")
        
        return sample


class JointRandomHorizontalFlip:
    """Horizontal-flip image + masks together."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            for key in ("image", "label", "inst", "scribble"):
                sample[key] = TF.hflip(sample[key])
        return sample


# -----------------------------------------------------------------------------
# Public transforms
# -----------------------------------------------------------------------------
class StrongAug:
    """(Stronger) data augmentation used in training.
    """

    def __init__(self, size: Tuple[int, int] | int) -> None:
        self.geom = T.Compose([
            JointRandomResizedCrop(size),
            JointRandomHorizontalFlip(p=0.5),
        ])
        self.photo = T.Compose([
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])

    def __call__(self, sample: Dict) -> Dict:
        sample = self.geom(sample)
        sample["image"] = self.photo(sample["image"])
        return sample


class Resize:
    """Deterministic resize (used in val/test)."""

    def __init__(self, size: Tuple[int, int] | int):
        self.size = size

    def __call__(self, sample: Dict) -> Dict:
        sample["image"] = TF.resize(sample["image"], self.size,
                                     interpolation=Image.BILINEAR)
        for key in ("label", "inst", "scribble"):
            sample[key] = TF.resize(sample[key], self.size,
                                    interpolation=Image.NEAREST)
        return sample


class ToTensorNormalize:
    """Convert PIL images to *torch.Tensor* and normalize them."""

    def __init__(self,
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict) -> Dict:
        sample["image"] = TF.to_tensor(sample["image"])
        sample["image"] = TF.normalize(sample["image"], self.mean, self.std)

        for key in ("label", "inst", "scribble"):
            sample[key] = torch.from_numpy(np.array(sample[key], dtype=np.uint8)).long()
        return sample