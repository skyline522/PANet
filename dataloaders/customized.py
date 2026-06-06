# """
# Customized dataset
# """

# import os
# import random

# import torch
# import numpy as np

# from .pascal import VOC
from .hanfeg import HanfegWeld
# from .coco import COCOSeg
# from .common import PairedDataset


# def attrib_basic(_sample, class_id):
#     """
#     Add basic attribute

#     Args:
#         _sample: data sample
#         class_id: class label asscociated with the data
#             (sometimes indicting from which subset the data are drawn)
#     """
#     return {'class_id': class_id}


# def getMask(label, scribble, class_id, class_ids):
#     """
#     Generate FG/BG mask from the segmentation mask

#     Args:
#         label:
#             semantic mask
#         scribble:
#             scribble mask
#         class_id:
#             semantic class of interest
#         class_ids:
#             all class id in this episode
#     """
#     # Dense Mask
#     fg_mask = torch.where(label == class_id,
#                           torch.ones_like(label), torch.zeros_like(label))
#     bg_mask = torch.where(label != class_id,
#                           torch.ones_like(label), torch.zeros_like(label))
#     for class_id in class_ids:
#         bg_mask[label == class_id] = 0

#     # Scribble Mask
#     bg_scribble = scribble == 0
#     fg_scribble = torch.where((fg_mask == 1)
#                               & (scribble != 0)
#                               & (scribble != 255),
#                               scribble, torch.zeros_like(fg_mask))
#     scribble_cls_list = list(set(np.unique(fg_scribble)) - set([0,]))
#     if scribble_cls_list:  # Still need investigation
#         fg_scribble = fg_scribble == random.choice(scribble_cls_list).item()
#     else:
#         fg_scribble[:] = 0

#     return {'fg_mask': fg_mask,
#             'bg_mask': bg_mask,
#             'fg_scribble': fg_scribble.long(),
#             'bg_scribble': bg_scribble.long()}


# def fewShot(paired_sample, n_ways, n_shots, cnt_query, coco=False):
#     """
#     Postprocess paired sample for fewshot settings

#     Args:
#         paired_sample:
#             data sample from a PairedDataset
#         n_ways:
#             n-way few-shot learning
#         n_shots:
#             n-shot few-shot learning
#         cnt_query:
#             number of query images for each class in the support set
#         coco:
#             MS COCO dataset
#     """
#     ###### Compose the support and query image list ######
#     cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query])

#     # support class ids
#     class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

#     # support images
#     support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
#                       for i in range(n_ways)]
#     support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)]
#                         for i in range(n_ways)]

#     # support image labels
#     if coco:
#         support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
#                            for j in range(n_shots)] for i in range(n_ways)]
#     else:
#         support_labels = [[paired_sample[cumsum_idx[i] + j]['label'] for j in range(n_shots)]
#                           for i in range(n_ways)]
#     support_scribbles = [[paired_sample[cumsum_idx[i] + j]['scribble'] for j in range(n_shots)]
#                          for i in range(n_ways)]
#     support_insts = [[paired_sample[cumsum_idx[i] + j]['inst'] for j in range(n_shots)]
#                      for i in range(n_ways)]



#     # query images, masks and class indices
#     query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
#                     for j in range(cnt_query[i])]
#     query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways)
#                       for j in range(cnt_query[i])]
#     if coco:
#         query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]]
#                         for i in range(n_ways) for j in range(cnt_query[i])]
#     else:
#         query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'] for i in range(n_ways)
#                         for j in range(cnt_query[i])]
#     query_cls_idx = [sorted([0,] + [class_ids.index(x) + 1
#                                     for x in set(np.unique(query_label)) & set(class_ids)])
#                      for query_label in query_labels]


#     ###### Generate support image masks ######
#     support_mask = [[getMask(support_labels[way][shot], support_scribbles[way][shot],
#                              class_ids[way], class_ids)
#                      for shot in range(n_shots)] for way in range(n_ways)]


#     ###### Generate query label (class indices in one episode, i.e. the ground truth)######
#     query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
#     for i, query_label_tmp in enumerate(query_labels_tmp):
#         query_label_tmp[query_labels[i] == 255] = 255
#         for j in range(n_ways):
#             query_label_tmp[query_labels[i] == class_ids[j]] = j + 1

#     ###### Generate query mask for each semantic class (including BG) ######
#     # BG class
#     query_masks = [[torch.where(query_label == 0,
#                                 torch.ones_like(query_label),
#                                 torch.zeros_like(query_label))[None, ...],]
#                    for query_label in query_labels]
#     # Other classes in query image
#     for i, query_label in enumerate(query_labels):
#         for idx in query_cls_idx[i][1:]:
#             mask = torch.where(query_label == class_ids[idx - 1],
#                                torch.ones_like(query_label),
#                                torch.zeros_like(query_label))[None, ...]
#             query_masks[i].append(mask)


#     return {'class_ids': class_ids,

#             'support_images_t': support_images_t,
#             'support_images': support_images,
#             'support_mask': support_mask,
#             'support_inst': support_insts,

#             'query_images_t': query_images_t,
#             'query_images': query_images,
#             'query_labels': query_labels_tmp,
#             'query_masks': query_masks,
#             'query_cls_idx': query_cls_idx,
#            }


# def voc_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
#                 n_queries=1):
#     """
#     Args:
#         base_dir:
#             VOC dataset directory
#         split:
#             which split to use
#             choose from ('train', 'val', 'trainval', 'trainaug')
#         transform:
#             transformations to be performed on images/masks
#         to_tensor:
#             transformation to convert PIL Image to tensor
#         labels:
#             object class labels of the data
#         n_ways:
#             n-way few-shot learning, should be no more than # of object class labels
#         n_shots:
#             n-shot few-shot learning
#         max_iters:
#             number of pairs
#         n_queries:
#             number of query images
#     """
#     voc = VOC(base_dir=base_dir, split=split, transforms=transforms, to_tensor=to_tensor)
#     voc.add_attrib('basic', attrib_basic, {})

#     # Load image ids for each class
#     sub_ids = []
#     for label in labels:
#         with open(os.path.join(voc._id_dir, voc.split,
#                                'class{}.txt'.format(label)), 'r') as f:
#             sub_ids.append(f.read().splitlines())
#     # Create sub-datasets and add class_id attribute
#     subsets = voc.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

#     # Choose the classes of queries
#     cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries), minlength=n_ways)
#     # Set the number of images for each class
#     n_elements = [n_shots + x for x in cnt_query]
#     # Create paired dataset
#     paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
#                                 pair_based_transforms=[
#                                     (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
#                                                'cnt_query': cnt_query})])
#     return paired_data


# def coco_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
#                  n_queries=1):
#     """
#     Args:
#         base_dir:
#             COCO dataset directory
#         split:
#             which split to use
#             choose from ('train', 'val')
#         transform:
#             transformations to be performed on images/masks
#         to_tensor:
#             transformation to convert PIL Image to tensor
#         labels:
#             labels of the data
#         n_ways:
#             n-way few-shot learning, should be no more than # of labels
#         n_shots:
#             n-shot few-shot learning
#         max_iters:
#             number of pairs
#         n_queries:
#             number of query images
#     """
#     cocoseg = COCOSeg(base_dir, split, transforms, to_tensor)
#     cocoseg.add_attrib('basic', attrib_basic, {})

#     # Load image ids for each class
#     cat_ids = cocoseg.coco.getCatIds()
#     sub_ids = [cocoseg.coco.getImgIds(catIds=cat_ids[i - 1]) for i in labels]
#     # Create sub-datasets and add class_id attribute
#     subsets = cocoseg.subsets(sub_ids, [{'basic': {'class_id': cat_ids[i - 1]}} for i in labels])

#     # Choose the classes of queries
#     cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries),
#                             minlength=n_ways)
#     # Set the number of images for each class
#     n_elements = [n_shots + x for x in cnt_query]
#     # Create paired dataset
#     paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
#                                 pair_based_transforms=[
#                                     (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
#                                                'cnt_query': cnt_query, 'coco': True})])
#     return paired_data



# 文件名: customized.py (最终修改版)
"""
Customized dataset
"""

import os
import random

import torch
import numpy as np

from .pascal import VOC
from .hanfeg import HanfegWeld
from .coco import COCOSeg
from .common import PairedDataset


def attrib_basic(_sample, class_id):
    """
    Add basic attribute

    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}


def getMask(label, scribble, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id_in_list in class_ids: # 避免与外部class_id变量冲突
        bg_mask[label == class_id_in_list] = 0

    # Scribble Mask
    bg_scribble = scribble == 0
    fg_scribble = torch.where((fg_mask == 1)
                              & (scribble != 0)
                              & (scribble != 255),
                              scribble, torch.zeros_like(fg_mask))
    scribble_cls_list = list(set(np.unique(fg_scribble)) - set([0,]))
    if scribble_cls_list:  # Still need investigation
        fg_scribble = fg_scribble == random.choice(scribble_cls_list).item()
    else:
        fg_scribble[:] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,
            'fg_scribble': fg_scribble.long(),
            'bg_scribble': bg_scribble.long()}


def fewShot(paired_sample, n_ways, n_shots, cnt_query, coco=False):
    """
    Postprocess paired sample for fewshot settings

    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
        coco:
            MS COCO dataset
    """
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query])

    # support class ids
    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

    # support images
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)]
                        for i in range(n_ways)]

    # support image labels
    if coco:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
                           for j in range(n_shots)] for i in range(n_ways)]
    else:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'] for j in range(n_shots)]
                          for i in range(n_ways)]
    support_scribbles = [[paired_sample[cumsum_idx[i] + j]['scribble'] for j in range(n_shots)]
                         for i in range(n_ways)]
    support_insts = [[paired_sample[cumsum_idx[i] + j]['inst'] for j in range(n_shots)]
                     for i in range(n_ways)]

    ###### Generate support image masks ######
    support_mask = [[getMask(support_labels[way][shot], support_scribbles[way][shot],
                             class_ids[way], class_ids)
                     for shot in range(n_shots)] for way in range(n_ways)]

    # query images, masks and class indices
    query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways)
                      for j in range(cnt_query[i])]
    if coco:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]]
                        for i in range(n_ways) for j in range(cnt_query[i])]
    else:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'] for i in range(n_ways)
                        for j in range(cnt_query[i])]

    query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[query_labels[i] == 255] = 255
        for j in range(n_ways):
            query_label_tmp[query_labels[i] == class_ids[j]] = j + 1

    ###### Generate query mask for each semantic class (including BG) ######
    query_masks = []
    query_cls_idx = []
    for i, q_label in enumerate(query_labels):
        current_query_masks = []
        current_query_cls_idx = []
        
        current_query_masks.append(torch.where(q_label == 0,
                                               torch.ones_like(q_label),
                                               torch.zeros_like(q_label))[None, ...])
        current_query_cls_idx.append(0)
        
        for j in range(n_ways):
            fg_class_id_for_this_way = class_ids[j] 
            mask = torch.where(q_label == fg_class_id_for_this_way,
                               torch.ones_like(q_label),
                               torch.zeros_like(q_label))[None, ...]
            current_query_masks.append(mask)
            current_query_cls_idx.append(j + 1)
        
        query_masks.append(current_query_masks)
        query_cls_idx.append(current_query_cls_idx)

    return {'class_ids': class_ids,

            'support_images_t': support_images_t,
            'support_images': support_images,
            'support_mask': support_mask,
            'support_inst': support_insts,

            'query_images_t': query_images_t,
            'query_images': query_images,
            'query_labels': query_labels_tmp,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
           }


def voc_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                n_queries=1):
    voc = VOC(base_dir=base_dir, split=split, transforms=transforms, to_tensor=to_tensor)
    voc.add_attrib('basic', attrib_basic, {})

    sub_ids = []
    for label in labels:
        with open(os.path.join(voc._id_dir, voc.split,
                               'class{}.txt'.format(label)), 'r') as f:
            sub_ids.append(f.read().splitlines())
    subsets = voc.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    if split in ("val", "val2017"):                 # 只验证集保持单类 Query
      cnt_query = np.zeros(n_ways, dtype=int)
      cnt_query[0] = n_queries
    else:                                           # 训练集可按你的多类逻辑
      cnt_query = np.ones(n_ways, dtype=int)      # ① 每类先 1 张
      extra = max(0, n_queries - n_ways)          # ② 还有多少张没分完
      for idx in np.random.choice(n_ways, extra):
        cnt_query[idx] += 1

    n_elements = [n_shots + x for x in cnt_query]
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query})])
    return paired_data


def coco_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                 n_queries=1):
    cocoseg = COCOSeg(base_dir, split, transforms, to_tensor)
    cocoseg.add_attrib('basic', attrib_basic, {})

    cat_ids = cocoseg.coco.getCatIds()
    sub_ids = [cocoseg.coco.getImgIds(catIds=cat_ids[i - 1]) for i in labels]
    subsets = cocoseg.subsets(sub_ids, [{'basic': {'class_id': cat_ids[i - 1]}} for i in labels])

    if split in ("val", "val2017"):                 # 只验证集保持单类 Query
      cnt_query = np.zeros(n_ways, dtype=int)
      cnt_query[0] = n_queries
    else:                                           # 训练集可按你的多类逻辑
      cnt_query = np.ones(n_ways, dtype=int)      # ① 每类先 1 张
      extra = max(0, n_queries - n_ways)          # ② 还有多少张没分完
      for idx in np.random.choice(n_ways, extra):
        cnt_query[idx] += 1 

    n_elements = [n_shots + x for x in cnt_query]
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query, 'coco': True})])
    return paired_data


def hanfeg_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                   n_queries=1):
    """
    Hanfeg/Weld few-shot episodic dataset factory.

    labels: list of class ids for current fold (e.g., [1,2,3,4] for four weld shapes)
    注意：如果你只做 1-way，那么 labels 仍然传入当前 fold 的 base/novel 类集合，
          由 PairedDataset 采样决定 episode 的 class_id。
    """
    ds = HanfegWeld(base_dir=base_dir, split=split, transforms=transforms, to_tensor=to_tensor)
    ds.add_attrib('basic', attrib_basic, {})

    # Try to load class-wise id lists if present; else build on-the-fly by scanning masks once.
    sub_ids = []
    # Pascal-like class-wise files (optional):
    #   ImageSets/Segmentation/{split}/class{label}.txt  or ImageSets/Segmentation/class{label}.txt
    for label in labels:
        txt1 = os.path.join(ds._id_dir, split, f'class{label}.txt')
        txt2 = os.path.join(ds._id_dir, f'class{label}.txt')
        if os.path.exists(txt1):
            with open(txt1, 'r') as f:
                sub_ids.append(f.read().splitlines())
        elif os.path.exists(txt2):
            with open(txt2, 'r') as f:
                sub_ids.append(f.read().splitlines())
        else:
            # build: scan all masks once
            # Here we fall back to "all indices" filtering by mask contents in subsets()
            sub_ids.append(list(range(len(ds))))

    subsets = ds.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    # cnt_query policy: follow VOC logic
    if split in ("val", "test"):
        cnt_query = np.zeros(n_ways, dtype=int)
        cnt_query[0] = n_queries
    else:
        cnt_query = np.ones(n_ways, dtype=int)
        extra = max(0, n_queries - n_ways)
        for i in range(extra):
            cnt_query[i % n_ways] += 1

    paired = PairedDataset(
        subsets,
        n_ways=n_ways,
        n_shots=n_shots,
        n_queries=cnt_query,
        max_iters=max_iters,
        same_way=False,
        shuffle=split not in ("val", "test"),
    )

    paired.add_attrib('basic', attrib_basic, {})
    return paired.map(lambda sample: fewShot(sample, n_ways, n_shots, cnt_query, coco=False))
