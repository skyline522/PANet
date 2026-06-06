from __future__ import annotations
import os
from typing import Dict

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config import ex
from dpanet_pml import DPANet_PML
from resnet_cbam import resnet50_plain, resnet50_cbam

from dataloaders.customized import voc_fewshot, coco_fewshot, hanfeg_fewshot
from dataloaders.transforms import Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from util.metric import Metric


def build_support_tensors(epi, n_ways: int, n_shots: int, device):
    sup_imgs = torch.cat([s for w in epi['support_images'] for s in w], 0).to(device)
    sup_fg = torch.cat([s['fg_mask'] for w in epi['support_mask'] for s in w], 0).gt(0).long()
    sup_masks = torch.zeros_like(sup_fg, dtype=torch.long)
    for w in range(n_ways):
        for s in range(n_shots):
            idx = w * n_shots + s
            sup_masks[idx] = sup_fg[idx] * (w + 1)
    return sup_imgs, sup_masks.to(device)


@ex.automain
def main(_config: Dict):
    assert _config['mode'] in ('test', 'train')

    set_seed(_config['seed'])
    device = torch.device(f"cuda:{_config['gpu_id']}" if torch.cuda.is_available() else 'cpu')

    backbone_name = _config['model'].get('backbone', 'plain').lower()
    if backbone_name == 'cbam':
        backbone = resnet50_cbam(pretrained_path=_config['path']['init_path'])
    else:
        backbone = resnet50_plain(pretrained_path=_config['path']['init_path'])

    model = DPANet_PML(
     backbone=backbone,
     n_way=_config['task']['n_ways'],
     n_parts=_config['model']['n_parts'],
     alpha=_config['model']['alpha'],
     tau=_config['model']['tau'],
     temperature=_config['model']['temperature'],
     use_tcc_head=_config['model'].get('use_tcc_head', True),
     tcc_fuse_skel=_config['model'].get('tcc_fuse_skel', 0.25),
     tcc_fuse_edge=_config['model'].get('tcc_fuse_edge', 0.15),
    ).to(device)

    ckpt_path = _config.get('ckpt_path', '')
    if not ckpt_path:
        raise ValueError("ckpt_path is empty. Please provide a checkpoint path.")

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=True)
        if 'step' in ckpt and 'miou' in ckpt:
            print(f"Loaded checkpoint: step={ckpt['step']}, saved_mIoU={ckpt['miou']:.4f}")
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()

    ds_name = _config['dataset'].upper()
    make_data = voc_fewshot if ds_name == 'VOC' else (coco_fewshot if ds_name == 'COCO' else hanfeg_fewshot)
    labels = CLASS_LABELS[_config['dataset']][_config['label_sets']] if _config['dataset'] in CLASS_LABELS else [1, 2, 3, 4]

    val_ds = make_data(
        base_dir=_config['path'][_config['dataset']]['data_dir'],
        split=_config['path'][_config['dataset']]['val_split'],
        transforms=Compose([Resize(size=_config['input_size']), ToTensorNormalize()]),
        to_tensor=None,
        labels=labels,
        max_iters=1000,
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries'],
    )
    loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=_config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=_config.get('num_workers', 4) > 0,
    )

    metric = Metric(max_label=_config['task']['n_ways'])

    with torch.no_grad():
        for epi in loader:
            sup_imgs, sup_masks = build_support_tensors(
                epi,
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                device=device,
            )
            qry_img = epi['query_images'][0].to(device)
            qry_lbl = epi['query_labels'][0].long().to(device)

            logits, _ = model(sup_imgs, sup_masks, qry_img)
            logits_up = F.interpolate(logits, size=qry_lbl.shape[-2:], mode='bilinear', align_corners=False)
            pred = logits_up.argmax(1)
            metric.update(pred, qry_lbl, ignore_label=_config['ignore_label'])

    miou, fb_iou = metric.compute_iou()
    print(f"mIoU={float(miou):.2f}  FB-IoU={float(fb_iou):.2f}")
