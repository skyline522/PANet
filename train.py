from __future__ import annotations
import os
from typing import Dict

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.amp import autocast, GradScaler

from config import ex
from dpanet_pml import DPANet_PML
from resnet_cbam import resnet50_plain, resnet50_cbam
from pml_loss import PixelMetricLoss

from dataloaders.customized import voc_fewshot, coco_fewshot, hanfeg_fewshot
from dataloaders.transforms import StrongAug, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from util.metric import Metric
from util.tcc_ops import make_edge_target, make_skeleton_target, balanced_bce_with_logits
from util.ldr_aug import random_photometric_aug


def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, eps: float = 1e-6):
    pred = torch.softmax(logits, dim=1)
    valid = (target != ignore_index)
    target_clamped = target.clone()
    target_clamped[~valid] = 0
    one_hot = F.one_hot(target_clamped, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    pred = pred * valid
    one_hot = one_hot * valid
    inter = (pred * one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def build_support_tensors(epi, n_ways: int, n_shots: int, device):
    sup_imgs = torch.cat([s for w in epi['support_images'] for s in w], 0).to(device)
    sup_fg = torch.cat([s['fg_mask'] for w in epi['support_mask'] for s in w], 0).gt(0).long()
    sup_masks = torch.zeros_like(sup_fg, dtype=torch.long)
    for w in range(n_ways):
        for s in range(n_shots):
            idx = w * n_shots + s
            sup_masks[idx] = sup_fg[idx] * (w + 1)
    return sup_imgs, sup_masks.to(device)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device, ignore_label: int, n_ways: int):
    model.eval()
    metric = Metric(max_label=n_ways)
    for epi in loader:
        sup_imgs, sup_masks = build_support_tensors(epi, n_ways=n_ways, n_shots=len(epi['support_images'][0]), device=device)
        qry_img = epi['query_images'][0].to(device)
        qry_lbl = epi['query_labels'][0].long().to(device)

        logits, _ = model(sup_imgs, sup_masks, qry_img)
        logits_up = F.interpolate(logits, size=qry_lbl.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits_up.argmax(1)
        metric.update(pred, qry_lbl, ignore_label=ignore_label)

    miou, fb_iou = metric.compute_iou()
    model.train()
    return float(miou), float(fb_iou)


@ex.automain
def main(_config: Dict):
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
    ).to(device).train()

    ds_name = _config['dataset'].upper()
    make_data = voc_fewshot if ds_name == 'VOC' else (coco_fewshot if ds_name == 'COCO' else hanfeg_fewshot)
    labels = CLASS_LABELS[_config['dataset']][_config['label_sets']] if _config['dataset'] in CLASS_LABELS else [1, 2, 3, 4]

    train_ds = make_data(
        base_dir=_config['path'][_config['dataset']]['data_dir'],
        split=_config['path'][_config['dataset']]['data_split'],
        transforms=Compose([StrongAug(size=_config['input_size'])]),
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries'],
    )
    trainloader = DataLoader(
        train_ds,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=_config.get('num_workers', 4) > 0,
    )

    val_ds = make_data(
        base_dir=_config['path'][_config['dataset']]['data_dir'],
        split=_config['path'][_config['dataset']]['val_split'],
        transforms=Compose([Resize(size=_config['input_size']), ToTensorNormalize()]),
        to_tensor=None,
        labels=labels,
        max_iters=500,
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries'],
    )
    valloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=_config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=_config.get('num_workers', 4) > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=_config['optim']['lr'], weight_decay=_config['optim']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=_config['optim']['lr'] ,
        total_steps=_config['n_steps'],
        pct_start=0.5,
        anneal_strategy='linear',
    )

    criterion_seg = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    criterion_epml = PixelMetricLoss(
        ignore_index=_config['ignore_label'],
        edge_radius=_config['loss']['edge_radius'],
        edge_weight=_config['loss']['edge_weight'],
        normalize=True,
    )
    ce_w = float(_config['loss']['ce_weight'])
    gamma = float(_config['loss']['epml_weight'])

    use_tcc = bool(_config['model'].get('use_tcc_head', True))
    tcc_skel_w = float(_config['loss'].get('tcc_skel_weight', 0.2))
    tcc_edge_w = float(_config['loss'].get('tcc_edge_weight', 0.2))
    tcc_edge_r = int(_config['loss'].get('tcc_edge_radius', 2))
    tcc_skel_r = int(_config['loss'].get('tcc_skel_radius', 1))
    tcc_iters = int(_config['loss'].get('tcc_skel_iters', 25))

    use_ldr = bool(_config.get('use_ldr', False))
    ldr_w = float(_config['loss'].get('ldr_weight', 0.0))

    scaler = GradScaler() if _config.get('use_amp', False) else None
    use_amp = _config.get('use_amp', False)
    val_interval = int(_config.get('val_interval', 2000))
    save_dir = os.path.join(_config['path']['log_dir'], f"FSS_Weld_Thesis_{_config['exp_str']}", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_miou = -1.0
    best_step = -1
    for step, epi in enumerate(trainloader, 1):
        sup_imgs, sup_masks = build_support_tensors(
            epi,
            n_ways=_config['task']['n_ways'],
            n_shots=_config['task']['n_shots'],
            device=device,
        )
        qry_imgs = epi['query_images'][0].to(device)
        qry_lbls = epi['query_labels'][0].long().to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            q_logits, aux = model(sup_imgs, sup_masks, qry_imgs)
            q_logits_up = F.interpolate(q_logits, size=qry_lbls.shape[-2:], mode='bilinear', align_corners=False)

            seg = ce_w * criterion_seg(q_logits_up, qry_lbls) + (1.0 - ce_w) * dice_loss(q_logits_up, qry_lbls, _config['ignore_label'])

            lbl_ds = F.interpolate(qry_lbls.unsqueeze(1).float(), size=q_logits.shape[-2:], mode='nearest').squeeze(1).long()
            epml = criterion_epml(q_logits, lbl_ds)

            skel_loss = q_logits.new_tensor(0.0)
            edge_loss = q_logits.new_tensor(0.0)
            if use_tcc and ('skel_logits' in aux) and ('edge_logits' in aux):
                valid = (lbl_ds != _config['ignore_label']).float().unsqueeze(1)
                skel_target = make_skeleton_target(lbl_ds, ignore_index=_config['ignore_label'], iters=tcc_iters, radius=tcc_skel_r)
                edge_target = make_edge_target(lbl_ds, ignore_index=_config['ignore_label'], radius=tcc_edge_r)
                skel_loss = balanced_bce_with_logits(aux['skel_logits'], skel_target, valid=valid)
                edge_loss = balanced_bce_with_logits(aux['edge_logits'], edge_target, valid=valid)

            ldr_loss = q_logits.new_tensor(0.0)
            if use_ldr and ldr_w > 0.0:
                qry_imgs_aug = random_photometric_aug(qry_imgs)
                q_logits_aug, _ = model(sup_imgs, sup_masks, qry_imgs_aug)
                teacher_prob = torch.softmax(q_logits.detach(), dim=1)
                student_log_prob = F.log_softmax(q_logits_aug, dim=1)
                valid = (lbl_ds != _config['ignore_label']).float()
                kl_map = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=1)
                ldr_loss = (kl_map * valid).sum() / valid.sum().clamp(min=1.0)

            total_loss = seg + gamma * epml + tcc_skel_w * skel_loss + tcc_edge_w * edge_loss + ldr_w * ldr_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[step {step:05d}] loss={float(total_loss):.4f} "
                f"seg={float(seg):.4f} epml={float(epml):.4f} "
                f"skel={float(skel_loss):.4f} edge={float(edge_loss):.4f} "
                f"ldr={float(ldr_loss):.4f} lr={lr:.6e}"
            )

        if step % val_interval == 0:
            miou, fb = validate(
                model,
                valloader,
                device,
                ignore_label=_config['ignore_label'],
                n_ways=_config['task']['n_ways'],
            )
            print(f"[val @ step {step:05d}] mIoU={miou:.2f} FB-IoU={fb:.2f}")

            ckpt = {
                'step': step,
                'miou': float(miou),
                'fb_iou': float(fb),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': dict(_config),
            }

            latest_path = os.path.join(save_dir, 'latest.pth')
            torch.save(ckpt, latest_path)

            if miou > best_miou:
                best_miou = float(miou)
                best_step = step
                best_path = os.path.join(save_dir, 'best.pth')
                torch.save(ckpt, best_path)
                print(f"[best] step={best_step:05d} mIoU={best_miou:.4f} saved to {best_path}")

        if step >= _config['n_steps']:
            break
    print(f"Training finished. Best mIoU={best_miou:.4f} at step={best_step}")
