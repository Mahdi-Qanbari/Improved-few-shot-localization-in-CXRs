from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import csv
import os

@HOOKS.register_module()
class LogClasswiseMAPHook(Hook):
    def __init__(self, csv_path='work_dirs/classwise_map.csv'):
        self.csv_path = csv_path
        self.logged_epochs = set()  # prevent duplicate logging
        print(f"[LogClasswiseMAPHook] Initialized with path edited: {csv_path}")

    def after_val_epoch(self, runner, metrics=None):
        print("[LogClasswiseMAPHook] after_val_epoch triggered")
        if metrics is None:
            runner.logger.warning('[LogClasswiseMAPHook] No metrics passed to hook. Skipping.')
            return
        self._log_to_csv(runner, metrics)

    def _log_to_csv(self, runner, metrics):
        print("[LogClasswiseMAPHook] _log_to_csv called")
        if metrics is None:
            runner.logger.warning('No metrics found. Skipping CSV log.')
            return

        current_epoch = runner.epoch
        if current_epoch in self.logged_epochs:
            print(f"[LogClasswiseMAPHook] Skipping duplicate log for epoch {current_epoch}")
            return  # skip duplicate logging
        self.logged_epochs.add(current_epoch)


        cleaned_metrics = {}
        for k, v in metrics.items():
            if k.startswith('coco/') and k.endswith('_precision'):
                clean_key = k[len('coco/'):-len('_precision')]
                cleaned_metrics[clean_key] = v
            elif k.startswith('coco/'):
                clean_key = k[len('coco/'):]
                cleaned_metrics[clean_key] = v

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        write_header = not os.path.exists(self.csv_path)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Epoch'] + list(cleaned_metrics.keys()))
            if write_header:
                writer.writeheader()
            row = {'Epoch': current_epoch}
            row.update({k: (round(v, 4) if v is not None else 'nan') for k, v in cleaned_metrics.items()})
            writer.writerow(row)

        runner.logger.info(f'[LogClasswiseMAPHook] mAPs written to edited: {self.csv_path}')
