import time
from typing import List, Optional, Union
import comet_ml

import torch
import tqdm
from torch.utils.data import DataLoader

from torchpack.callbacks.callback import Callback, Callbacks
from torchpack.callbacks import InferenceRunner
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from torchpack.utils.typing import Trainer
from tools.utils import augment, augment_merge

import os
import numpy as np

from torchpack.environ import get_run_dir
from torchpack.utils import fs
from torchpack.callbacks.writers import SummaryWriter


__all__ = ['MultiInferenceRunner', 'CometWriter']


class MultiInferenceRunner(InferenceRunner):
    def __init__(self, dataflow: DataLoader, *,
                 callbacks: List[Callback],
                 augment_setting: dict) -> None:
        super().__init__(dataflow, callbacks=callbacks)
        self.augment_setting = augment_setting

    def _trigger(self) -> None:
        start_time = time.perf_counter()
        self.callbacks.before_epoch()

        with torch.no_grad():
            for feed_dict in tqdm.tqdm(self.dataflow, ncols=0):
                self.callbacks.before_step(feed_dict)
                feed_dict = augment(feed_dict, self.augment_setting)
                output_dict = self.trainer.run_step(feed_dict)
                output_dict = augment_merge(output_dict, self.augment_setting)
                self.callbacks.after_step(output_dict)

        self.callbacks.after_epoch()
        logger.info('Inference finished in {}.'.format(
            humanize.naturaldelta(time.perf_counter() - start_time)))


class CometWriter(SummaryWriter):
    """
    Write summaries to Comet
    """
    def __init__(self, *, save_dir: Optional[str] = None,
                 configs: Optional[dict] = None,
                 project_name: Optional[str] = None) -> None:
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'tensorboard')
        self.save_dir = fs.normpath(save_dir)
        self.configs = configs
        self.project_name = project_name

    def _set_trainer(self, trainer: Trainer) -> None:
        from tensorboardX import SummaryWriter
        comet_config = {
            "api_key": os.environ["COMET_API_KEY"],
            "project_name": "healthcare-dev",
            "disabled": False
        }
        if self.project_name is not None:
            comet_config["project_name"] = self.project_name
        self.writer = SummaryWriter(log_dir=self.save_dir, comet_config=comet_config)
        if self.configs is not None:
            hparams=dict(self.configs)
            hparams['run'] = self.save_dir
            self.writer.add_hparams(
                hparam_dict=hparams,
                metric_dict={},
            )

    def _add_scalar(self, name: str, scalar: Union[int, float]) -> None:
        self.writer.add_scalar(name, scalar, self.trainer.global_step)

    def _add_image(self, name: str, tensor: np.ndarray) -> None:
        self.writer.add_image(name, tensor, self.trainer.global_step)

    def _after_train(self) -> None:
        os.environ["COMET_EXPERIMENT_KEY"] = comet_ml.get_global_experiment().get_key()
        os.environ["COMET_EXPERIMENT_NAME"] = comet_ml.get_global_experiment().get_name()
        tags = [self.configs.model.name, f'drop{self.configs.model.dropout}', f'out{self.configs.model.out_ch}',
                f'wd{self.configs.optimizer.weight_decay}', f'lr{self.configs.optimizer.lr}',
                self.configs.scheduler.name, f'layer{self.configs.model.layer_num}', f'batch{self.configs.batch_size}',
                f'fold{self.configs.dataset.fold}', f'seed{self.configs.dataset.seed}', f'{self.configs.device}',
                f'target{self.configs.dataset.target}']
        if self.configs.model.name == 'attn':
            tags.append(f'head{self.configs.model.n_head}')
        elif self.configs.model.name == 'lstm':
            tags.append('bd' if self.configs.model.bidirectional else 'ud')
        elif self.configs.model.name == 'conv':
            tags.append(f'kernel{self.configs.model.kernel_size}')
            tags.append(f'stride{self.configs.model.stride}')
            tags.append(f'pad{self.configs.model.padding}')
        for feat in self.configs.model.feats:
            tags.append(f'{feat}+')
        comet_ml.get_global_experiment().add_tags(tags)
        self.writer.close()
