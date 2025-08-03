from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import torch
import csv

# Torch's SummaryWriter does not support graphing hparams
# Solution from:
# https://discuss.pytorch.org/t/how-to-add-graphs-to-hparams-in-tensorboard/109349/2
class CustomWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None):
        self.metrics = list(metric_dict.keys())
        with open(f"{self.log_dir}/metrics.csv", 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["index"]+self.metrics)
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
    
    # replaces add_scalars as the original one was useless
    def add_scalars(self, params, global_step=None):
        assert set(params.keys()) == set(self.metrics), \
            "params keys must match the metrics keys defined in add_hparams"
        with open(f"{self.log_dir}/metrics.csv", 'a') as f:
            csv_writer = csv.writer(f)
            row = [global_step] + [params[k] for k in self.metrics]
            csv_writer.writerow(row)
        for key, value in params.items():
            self.add_scalar(key, value, global_step=global_step)
