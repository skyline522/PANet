
"""
hanfeg.py — Hanfeg / Weld seam dataset loader (for Chapter 4)

目标：让你的焊缝数据集也能走 few-shot episodic 采样范式（1-way/5-shot）。
本实现尽量“结构自适应”，支持两种常见组织方式：

A) Pascal-like
   data_dir/
     JPEGImages/*.jpg|png
     SegmentationClass/*.png
     ImageSets/Segmentation/train.txt  val.txt  test.txt
   mask 像素值：0=背景，1..N=类别（焊缝形态）

B) Simple folders
   data_dir/
     images/train/*.jpg|png, images/val/*.jpg|png
     masks/train/*.png, masks/val/*.png
   或 images/{split}/..., masks/{split}/...

你只需要保证“image 与 mask basename 对齐”即可。
"""

from __future__ import annotations
import os, glob
import numpy as np
from PIL import Image
import torch

from .common import BaseDataset


def _list_ids_from_txt(txt_path: str):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        ids = [x.strip() for x in f.read().splitlines() if x.strip()]
    return ids


def _guess_ext(path_no_ext: str):
    for ext in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]:
        if os.path.exists(path_no_ext + ext):
            return ext
    return None


class HanfegWeld(BaseDataset):
    """
    Return sample dict compatible with transforms.py / customized.py:
      sample["image"]   : PIL.Image (RGB)
      sample["label"]   : PIL.Image (mode='L', uint8)  0=bg, 1..C=classes, 255=ignore
      sample["scribble"]: PIL.Image (all zeros)  (placeholder)
      sample["inst"]    : PIL.Image (all zeros)  (placeholder)
    """
    def __init__(self, base_dir: str, split: str, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        self.transforms = transforms
        self.to_tensor = to_tensor

        # A) Pascal-like
        imgset_txt = os.path.join(self._base_dir, "ImageSets", "Segmentation", f"{split}.txt")
        pascal_img_dir = os.path.join(self._base_dir, "JPEGImages")
        pascal_lbl_dir = os.path.join(self._base_dir, "SegmentationClass")

        # B) folder-like
        img_dir_candidates = [
            os.path.join(self._base_dir, "images", split),
            os.path.join(self._base_dir, "Images", split),
            os.path.join(self._base_dir, split, "images"),
            os.path.join(self._base_dir, split, "Images"),
        ]
        msk_dir_candidates = [
            os.path.join(self._base_dir, "masks", split),
            os.path.join(self._base_dir, "Masks", split),
            os.path.join(self._base_dir, split, "masks"),
            os.path.join(self._base_dir, split, "Masks"),
            os.path.join(self._base_dir, "labels", split),
            os.path.join(self._base_dir, "Labels", split),
        ]

        self._mode = None
        self._ids = []
        self._img_paths = []
        self._lbl_paths = []

        if os.path.exists(imgset_txt) and os.path.isdir(pascal_img_dir) and os.path.isdir(pascal_lbl_dir):
            self._mode = "pascal"
            self._ids = _list_ids_from_txt(imgset_txt)
            for _id in self._ids:
                img_no_ext = os.path.join(pascal_img_dir, _id)
                ext = _guess_ext(img_no_ext)
                if ext is None:
                    # fallback: glob
                    cands = glob.glob(img_no_ext + ".*")
                    if not cands:
                        continue
                    img_path = cands[0]
                else:
                    img_path = img_no_ext + ext
                lbl_path = os.path.join(pascal_lbl_dir, _id + ".png")
                if not os.path.exists(lbl_path):
                    continue
                self._img_paths.append(img_path)
                self._lbl_paths.append(lbl_path)

        else:
            # folder mode
            img_dir = next((d for d in img_dir_candidates if os.path.isdir(d)), None)
            msk_dir = next((d for d in msk_dir_candidates if os.path.isdir(d)), None)
            if img_dir is None or msk_dir is None:
                raise FileNotFoundError(
                    f"Cannot locate Hanfeg/Weld dataset folders. Tried Pascal-like and {img_dir_candidates} / {msk_dir_candidates}"
                )
            self._mode = "folder"
            img_paths = sorted(sum([glob.glob(os.path.join(img_dir, f"*{e}")) for e in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]], []))
            for ip in img_paths:
                stem = os.path.splitext(os.path.basename(ip))[0]
                mp = os.path.join(msk_dir, stem + ".png")
                if os.path.exists(mp):
                    self._img_paths.append(ip)
                    self._lbl_paths.append(mp)

        if len(self._img_paths) == 0:
            raise RuntimeError(f"No image/mask pairs found under {base_dir} (split={split}).")

        # minimal id_dir for compatibility with customized.py (if you choose to precompute class lists)
        self._id_dir = os.path.join(self._base_dir, "ImageSets", "Segmentation")

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self._img_paths[idx]).convert("RGB")
        lbl = Image.open(self._lbl_paths[idx]).convert("L")
        # placeholders
        zeros = Image.fromarray(np.zeros((lbl.size[1], lbl.size[0]), dtype=np.uint8), mode="L")
        sample = {"image": img, "label": lbl, "scribble": zeros, "inst": zeros}

        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        return sample
