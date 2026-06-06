"""
Metrics for computing evaluation results.
Compatible with both old API (record/get_mIoU/get_mIoU_binary)
and new API used in your train.py/test.py (update/compute_iou).
"""

import numpy as np
import torch


class Metric(object):
    def __init__(self, max_label=20, n_runs=None):
        self.labels = list(range(max_label + 1))
        self.n_runs = 1 if n_runs is None else n_runs
        self.epsilon = 1e-9
        self.tp_lst = [[] for _ in range(self.n_runs)]
        self.fp_lst = [[] for _ in range(self.n_runs)]
        self.fn_lst = [[] for _ in range(self.n_runs)]

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    def record(self, pred, target, labels=None, n_run=None, ignore_label=255):
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        assert pred.shape == target.shape

        if self.n_runs == 1:
            n_run = 0
        if labels is None:
            labels = self.labels
        else:
            labels = [0] + list(labels)

        tp_arr = np.full(len(self.labels), np.nan, dtype=np.float64)
        fp_arr = np.full(len(self.labels), np.nan, dtype=np.float64)
        fn_arr = np.full(len(self.labels), np.nan, dtype=np.float64)

        valid = (target != ignore_label)
        for label in labels:
            pred_j = (pred == label) & valid
            target_j = (target == label) & valid
            if target_j.any() or pred_j.any():
                tp_arr[label] = np.logical_and(pred_j, target_j).sum()
                fp_arr[label] = np.logical_and(pred_j, ~target_j).sum()
                fn_arr[label] = np.logical_and(~pred_j, target_j).sum()

        self.tp_lst[n_run].append(tp_arr)
        self.fp_lst[n_run].append(fp_arr)
        self.fn_lst[n_run].append(fn_arr)

    def update(self, pred, target, ignore_label=255, labels=None, n_run=None):
        self.record(pred, target, labels=labels, n_run=n_run, ignore_label=ignore_label)

    def get_mIoU(self, labels=None, n_run=None):
        if labels is None:
            labels = self.labels
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels) for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels) for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels) for run in range(self.n_runs)]
            mIoU_class = np.vstack([
                tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run] + self.epsilon)
                for run in range(self.n_runs)
            ])
            mIoU = np.nanmean(mIoU_class, axis=1)
            return (np.nanmean(mIoU_class, axis=0), np.nanstd(mIoU_class, axis=0),
                    np.nanmean(mIoU, axis=0), np.nanstd(mIoU, axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum + self.epsilon)
            mIoU = np.nanmean(mIoU_class)
            return mIoU_class, mIoU

    def get_mIoU_binary(self, n_run=None):
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0) for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0) for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0) for run in range(self.n_runs)]
            tp_sum = [np.c_[tp_sum[run][0], np.nansum(tp_sum[run][1:])] for run in range(self.n_runs)]
            fp_sum = [np.c_[fp_sum[run][0], np.nansum(fp_sum[run][1:])] for run in range(self.n_runs)]
            fn_sum = [np.c_[fn_sum[run][0], np.nansum(fn_sum[run][1:])] for run in range(self.n_runs)]
            mIoU_class = np.vstack([
                tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run] + self.epsilon)
                for run in range(self.n_runs)
            ])
            mIoU = np.nanmean(mIoU_class, axis=1)
            return (np.nanmean(mIoU_class, axis=0), np.nanstd(mIoU_class, axis=0),
                    np.nanmean(mIoU, axis=0), np.nanstd(mIoU, axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0)
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum + self.epsilon)
            mIoU = np.nanmean(mIoU_class)
            return mIoU_class, mIoU

    def compute_iou(self, n_run=None):
        if n_run is None:
            n_run = 0
        _, miou = self.get_mIoU(n_run=n_run)
        _, fb_iou = self.get_mIoU_binary(n_run=n_run)
        return float(miou), float(fb_iou)
