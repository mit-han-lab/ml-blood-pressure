from typing import Any, Callable, Dict

import torch
from torch import nn
import torch.nn.functional as F
from torchpack.utils.config import configs
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler


__all__ = ['MAPTrainer', 'PPG2BPTrainer', 'BP2CVPTrainer',
           'MAPAdversarialTrainer',
           'MAPContrastiveTrainer']


class MAPTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.acc = []

    def _before_epoch(self) -> None:
        self.model.train()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        inputs = dict()
        for key, value in feed_dict.items():
            if key in ['area', 'v', 'pwv', 'age', 'comp', 'z0', 'deltat', 'pp',
                       'id', 'bp', 'flow', 'shape', 'weight', 'height', 'gender', 'heartrate', 
                       'diameter_complete_avg_beats', 'velocity_complete_avg_beats',
                       'bp_shape_complete_avg_beats', 'area_complete_avg_beats',
                       'bp_shape_complete_min', 'bp_shape_complete_mean', 'bp_shape_complete_max',
                       'velocity_complete_min', 'velocity_complete_mean', 'velocity_complete_max']:
                if configs.device == 'gpu':
                    inputs[key] = value.cuda()
                else:
                    inputs[key] = value
        targets = feed_dict[configs.dataset.target]
        if configs.device == 'gpu':
            targets = targets.cuda(non_blocking=True)
        else:
            pass
        if configs.model.name == 'attn':
            outputs, _ = self.model(inputs)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if loss.requires_grad:
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.scheduler.step()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])


def pw_cosine_distance(input_a, input_b):
    normalized_input_a = torch.nn.functional.normalize(input_a)
    normalized_input_b = torch.nn.functional.normalize(input_b)
    res = torch.mm(normalized_input_a, normalized_input_b.T)
    # res *= -1 # 1-res without copy
    # res += 1
    return res

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class MAPContrastiveTrainer(MAPTrainer):
    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        inputs = dict()
        for key, value in feed_dict.items():
            if key in ['a', 'v', 'pwv', 'age', 'comp', 'z0', 'deltat', 'pp',
                       'bp', 'flow']:
                if configs.device == 'gpu':
                    inputs[key] = value.cuda()
                else:
                    inputs[key] = value
        targets_map = feed_dict['map']
        targets_id = feed_dict['id']
        if configs.device == 'gpu':
            targets_map = targets_map.cuda(non_blocking=True)
            targets_id = targets_id.cuda(non_blocking=True)
        else:
            pass
        if 'attn' in configs.model.name:
            outputs, _ = self.model(inputs)
        else:
            outputs = self.model(inputs)

        outputs_map, outputs_contrastive_vectors = outputs[0], outputs[1]

        n_samples = int(outputs_map.shape[0] // 2)

        loss_map = self.criterion(outputs_map, targets_map)

        targets_contrastive = torch.LongTensor(list(range(n_samples))).to(outputs_map.device)
        # print(feed_dict['id'])

        outputs_contra1 = outputs_contrastive_vectors[:n_samples]
        outputs_contra2 = outputs_contrastive_vectors[n_samples:]
        outputs_dis = pw_cosine_distance(outputs_contra1, outputs_contra2)

        loss_contrastive = F.nll_loss(F.log_softmax(outputs_dis), targets_contrastive)

        loss = loss_map + loss_contrastive
        # print(f"loss_map {loss_map}, loss_id {loss_id}")

        if loss.requires_grad:
            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('loss_map', loss_map.item())
            self.summary.add_scalar('loss_contrastive', loss_contrastive.item())
            acc = accuracy(outputs_dis, targets_contrastive)[0].item()
            self.summary.add_scalar('contrastive_acc', acc)
            self.acc.append(acc)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs_map, 'targets': targets_map}


class MAPAdversarialTrainer(MAPTrainer):
    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        inputs = dict()
        for key, value in feed_dict.items():
            if key in ['a', 'v', 'pwv', 'age', 'comp', 'z0', 'deltat', 'pp',
                    'bp', 'flow']:
                if configs.device == 'gpu':
                    inputs[key] = value.cuda()
                else:
                    inputs[key] = value
        targets_map = feed_dict['map']
        targets_id = feed_dict['id']
        if configs.device == 'gpu':
            targets_map = targets_map.cuda(non_blocking=True)
            targets_id = targets_id.cuda(non_blocking=True)
        else:
            pass
        if 'attn' in configs.model.name:
            outputs, _ = self.model(inputs)
        else:
            outputs = self.model(inputs)

        outputs_map, outputs_id = outputs[0], outputs[1]
        loss_map = self.criterion(outputs_map, targets_map)
        loss_id = F.nll_loss(outputs_id, targets_id)
        loss = loss_map + loss_id
        # print(f"loss_map {loss_map}, loss_id {loss_id}")


        if loss.requires_grad:
            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('loss_map', loss_map.item())
            self.summary.add_scalar('loss_id', loss_id.item())
            acc = accuracy(outputs_id, targets_id)[0].item()
            self.summary.add_scalar('id accuracy', acc)
            self.acc.append(acc)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs_map, 'targets': targets_map}


class PPG2BPTrainer(MAPTrainer):
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler) -> None:
        super().__init__(model, criterion, optimizer, scheduler)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _before_epoch(self) -> None:
        self.model.train()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        inputs = dict()
        for key, value in feed_dict.items():
            if key in ['id', 'ppg', 'age', 'sex', 'height', 'weight', 'bmi']:
                if configs.device == 'gpu':
                    inputs[key] = value.cuda()
                else:
                    inputs[key] = value
        targets = feed_dict['bp'].float()

        if configs.device == 'gpu':
            targets = targets.cuda(non_blocking=True)
        else:
            pass

        if configs.model.name == 'attn':
            outputs, _ = self.model(inputs)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if loss.requires_grad:
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}


class BP2CVPTrainer(MAPTrainer):
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler) -> None:
        super().__init__(model, criterion, optimizer, scheduler)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _before_epoch(self) -> None:
        self.model.train()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        inputs = dict()
        for key, value in feed_dict.items():
            if key in ['bp', 'age', 'sex', 'height', 'weight', 'bmi']:
                if configs.device == 'gpu':
                    inputs[key] = value.cuda()
                else:
                    inputs[key] = value
        targets = feed_dict['cvp'].float()

        if configs.device == 'gpu':
            targets = targets.cuda(non_blocking=True)
        else:
            pass

        if configs.model.name == 'attn':
            outputs, _ = self.model(inputs)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if loss.requires_grad:
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}
