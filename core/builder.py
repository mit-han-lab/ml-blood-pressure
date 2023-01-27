import torch
import torch.nn
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Optimizer, Scheduler

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


def make_dataset(ensembled, mean=None, std=None, part=None, split_ratio=None):
    if configs.dataset.name == 'pwdb':
        from core.datasets import Pwdb
        dataset = Pwdb(root=configs.dataset.root,
                       split_ratio=configs.dataset.split_ratio,
                       location=configs.dataset.location,
                       resample_len=configs.dataset.resample_len,
                       augment_setting=configs.dataset.augment_setting)
    elif configs.dataset.name == 'js':
        from core.datasets import Js
        dataset = Js(root=configs.dataset.root,
                     split_ratio=configs.dataset.split_ratio,
                     resample_len=configs.dataset.resample_len,
                     augment_setting=configs.dataset.augment_setting)
    elif configs.dataset.name == 'pwdb_js':
        from core.datasets import PwdbJs
        dataset = PwdbJs(root_pwdb=configs.dataset.root_pwdb,
                         root_js=configs.dataset.root_js,
                         split_ratio_pwdb=configs.dataset.split_ratio_pwdb,
                         split_ratio_js=configs.dataset.split_ratio_js,
                         resample_len=configs.dataset.resample_len,
                         location_pwdb=configs.dataset.location_pwdb,
                         augment_setting=configs.dataset.augment_setting,
                         sample_prob=configs.dataset.sample_prob)
    elif configs.dataset.name == 'pwdb_pert':
        from core.datasets import PwdbPert
        dataset = PwdbPert(
            root=configs.dataset.root,
            split_ratio=configs.dataset.split_ratio,
            location=configs.dataset.location,
            resample_len=configs.dataset.resample_len,
            augment_setting=configs.dataset.augment_setting
        )
    elif configs.dataset.name == 'pwdb_measured_v1':
        from core.datasets import PwdbMeasuredV1
        dataset = PwdbMeasuredV1(root_pwdb=configs.dataset.root_pwdb,
                       root_measured_v1=configs.dataset.root_measured_v1,
                       split_ratio=configs.dataset.split_ratio,
                       location=configs.dataset.location,
                       resample_len=configs.dataset.resample_len,
                       augment_setting=configs.dataset.augment_setting)
    elif configs.dataset.name == 'pwdb_measured_v1_mix':
        from core.datasets import PwdbMeasuredV1Mix
        dataset = PwdbMeasuredV1Mix(root_pwdb=configs.dataset.root_pwdb,
                                 root_measured_v1=configs.dataset.root_measured_v1,
                                 split_ratio=configs.dataset.split_ratio,
                                 location=configs.dataset.location,
                                 resample_len=configs.dataset.resample_len,
                                 augment_setting=configs.dataset.augment_setting,
                                 normalize_a_each_beat=configs.dataset.normalize_a_each_beat,
                                 n_cross_validation_fold=configs.dataset.n_cross_validation_fold,
                                 cross_validation_fold_idx=configs.dataset.cross_validation_fold_idx,
                                )
    elif configs.dataset.name == 'measured_v2':
        from core.datasets import MeasuredV2
        dataset = MeasuredV2(root=configs.dataset.root,
                             split_ratio=configs.dataset.split_ratio,
                             beats_per_subject=configs.dataset.beats_per_subject,
                             subject_name=getattr(configs.dataset, 'subject_name', None)
                             )
    elif configs.dataset.name == 'measured_mit_v1':
        from core.datasets import MeasuredMITV1
        dataset = MeasuredMITV1(root=configs.dataset.root,
                             split_ratio=configs.dataset.split_ratio if split_ratio is None else split_ratio,
                             custom_split_indices=configs.dataset.custom_split_indices,
                             beats_per_subject=configs.dataset.beats_per_subject,
                             subject_name=getattr(configs.dataset, 'subject_name', None),
                             target=getattr(configs.dataset, 'target', 'map'),
                             part=configs.dataset.part if part is None else part,
                             mean=mean,
                             std=std,
                             ensembled=ensembled,
                             )
    elif configs.dataset.name == 'measured_v2_contrastive':
        from core.datasets import MeasuredV2Contrastive
        dataset = MeasuredV2Contrastive(root=configs.dataset.root,
                             split_ratio=configs.dataset.split_ratio,
                             beats_per_subject=configs.dataset.beats_per_subject,
                             subject_name=getattr(configs.dataset, 'subject_name', None)
                             )
    elif configs.dataset.name == 'ppg2bp_single':
        from core.datasets import PPG2BPSingle
        dataset = PPG2BPSingle(
            root=configs.dataset.root,
            split_ratio=configs.dataset.split_ratio,
            location=configs.dataset.location,
            resample_len=configs.dataset.resample_len,
            augment_setting=configs.dataset.augment_setting
        )
    elif configs.dataset.name == 'ppg2bp':
        from core.datasets import PPG2BP
        dataset = PPG2BP(
            root=configs.dataset.root,
            split_ratio=configs.dataset.split_ratio,
            location=configs.dataset.location,
            resample_len=configs.dataset.resample_len,
            augment_setting=configs.dataset.augment_setting,
            n_beats=configs.dataset.n_beats,
            across_all=configs.dataset.across_all
        )
    elif configs.dataset.name == 'bp2cvp':
        from core.datasets import BP2CVP
        dataset = BP2CVP(
            root=configs.dataset.root,
            split_ratio=configs.dataset.split_ratio,
            location=configs.dataset.location,
            resample_len=configs.dataset.resample_len,
            augment_setting=configs.dataset.augment_setting,
            n_beats=configs.dataset.n_beats,
        )
    elif configs.dataset.name == 'bp2cvp_single':
        from core.datasets import BP2CVPSingle
        dataset = BP2CVPSingle(
            root=configs.dataset.root,
            split_ratio=configs.dataset.split_ratio,
            location=configs.dataset.location,
            resample_len=configs.dataset.resample_len,
            augment_setting=configs.dataset.augment_setting,
            subject_id=configs.dataset.subject_id
        )
    elif configs.dataset.name == 'bp2cvp_common':
        from core.datasets import BP2CVPCommon
        dataset = BP2CVPCommon(
            root=configs.dataset.root,
            split_ratio=configs.dataset.split_ratio,
            location=configs.dataset.location,
            resample_len=configs.dataset.resample_len,
            augment_setting=configs.dataset.augment_setting,
            n_beats=configs.dataset.n_beats,
        )
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


def make_model() -> nn.Module:
    if configs.model.name == 'fc':
        from core.models import FC
        model = FC(in_ch=configs.model.in_ch,
                   out_ch=configs.model.out_ch,
                   layer_num=configs.model.layer_num,
                   dropout=configs.model.dropout,)
    elif configs.model.name == 'fc_2d':
        from core.models import FC2d
        model = FC2d(in_ch=configs.model.in_ch,
                     out_ch=configs.model.out_ch,
                     layer_num=configs.model.layer_num)
    elif configs.model.name == 'attn':
        from core.models import Attn
        model = Attn(in_ch=configs.model.in_ch,
                     out_ch=configs.model.out_ch,
                     n_head=configs.model.n_head,
                     layer_num=configs.model.layer_num,
                     dropout=configs.model.dropout,)
    elif configs.model.name == 'attn_adv':
        from core.models import AttnAdversarial
        model = AttnAdversarial(in_ch=configs.model.in_ch,
                     out_ch=configs.model.out_ch,
                     n_head=configs.model.n_head,
                     layer_num=configs.model.layer_num,
                                n_class=configs.model.n_class)
    elif configs.model.name == 'attn_contrastive':
        from core.models import AttnContrastive
        model = AttnContrastive(in_ch=configs.model.in_ch,
                                out_ch=configs.model.out_ch,
                                n_head=configs.model.n_head,
                                layer_num=configs.model.layer_num,
                                n_class=configs.model.n_class)
    elif configs.model.name == 'attn_torch':
        from core.models import AttnTorch
        model = AttnTorch(in_ch=configs.model.in_ch,
                          out_ch=configs.model.out_ch,
                          n_head=configs.model.n_head,
                          layer_num=configs.model.layer_num)
    elif configs.model.name == 'conv':
        from core.models import Conv
        model = Conv(in_ch=configs.model.in_ch,
                     out_ch=configs.model.out_ch,
                     kernel_size=configs.model.kernel_size,
                     stride=configs.model.stride,
                     padding=configs.model.padding,
                     layer_num=configs.model.layer_num,
                     dropout=configs.model.dropout)
    elif configs.model.name == 'conv_attn_torch':
        from core.models import ConvAttnTorch
        model = ConvAttnTorch(in_ch=configs.model.in_ch,
                              out_ch=configs.model.out_ch,
                              kernel_size=configs.model.kernel_size,
                              stride=configs.model.stride,
                              padding=configs.model.padding,
                              n_head=configs.model.n_head,
                              conv_layer_num=configs.model.conv_layer_num,
                              attn_layer_num=configs.model.attn_layer_num)
    elif configs.model.name == 'lstm':
        from core.models import Lstm
        model = Lstm(in_ch=configs.model.in_ch,
                     out_ch=configs.model.out_ch,
                     bidirectional=configs.model.bidirectional,
                     layer_num=configs.model.layer_num,
                     dropout=configs.model.dropout)
    elif configs.model.name == 'attn_seq':
        from core.models import AttnSeq
        model = AttnSeq(in_ch=configs.model.in_ch,
                        out_ch=configs.model.out_ch,
                        n_head=configs.model.n_head,
                        layer_num=configs.model.layer_num)
    elif configs.model.name == 'conv_seq':
        from core.models import ConvSeq
        model = ConvSeq(in_ch=configs.model.in_ch,
                        out_ch=configs.model.out_ch,
                        kernel_size=configs.model.kernel_size,
                        stride=configs.model.stride,
                        padding=configs.model.padding,
                        layer_num=configs.model.layer_num)
    elif configs.model.name == 'conv_attn_torch_seq':
        from core.models import ConvAttnTorchSeq
        model = ConvAttnTorchSeq(in_ch=configs.model.in_ch,
                                 out_ch=configs.model.out_ch,
                                 kernel_size=configs.model.kernel_size,
                                 stride=configs.model.stride,
                                 padding=configs.model.padding,
                                 n_head=configs.model.n_head,
                                 conv_layer_num=configs.model.conv_layer_num,
                                 attn_layer_num=configs.model.attn_layer_num)

    elif configs.model.name == 'conv_attn_torch_seq_anthro':
        from core.models import ConvAttnTorchSeqAnthro
        model = ConvAttnTorchSeqAnthro(
            in_ch=configs.model.in_ch,
            out_ch=configs.model.out_ch,
            kernel_size=configs.model.kernel_size,
            stride=configs.model.stride,
            padding=configs.model.padding,
            n_head=configs.model.n_head,
            conv_layer_num=configs.model.conv_layer_num,
            attn_layer_num=configs.model.attn_layer_num)
    elif configs.model.name == 'conv_attn_torch_bp2cvp':
        from core.models import ConvAttnTorchBP2CVP
        model = ConvAttnTorchBP2CVP(
            in_ch=configs.model.in_ch,
            out_ch=configs.model.out_ch,
            kernel_size=configs.model.kernel_size,
            stride=configs.model.stride,
            padding=configs.model.padding,
            n_head=configs.model.n_head,
            conv_layer_num=configs.model.conv_layer_num,
            attn_layer_num=configs.model.attn_layer_num)
    elif configs.model.name == 'conv_attn_torch_bp2cvp_single':
        from core.models import ConvAttnTorchBP2CVPSingle
        model = ConvAttnTorchBP2CVPSingle(
            in_ch=configs.model.in_ch,
            out_ch=configs.model.out_ch,
            kernel_size=configs.model.kernel_size,
            stride=configs.model.stride,
            padding=configs.model.padding,
            n_head=configs.model.n_head,
            conv_layer_num=configs.model.conv_layer_num,
            attn_layer_num=configs.model.attn_layer_num)
    elif configs.model.name == 'fc_bp2cvp_single':
        from core.models import FC_bp2cvp_single
        model = FC_bp2cvp_single(in_ch=configs.model.in_ch,
                                 out_ch=configs.model.out_ch,
                                 layer_num=configs.model.layer_num)
    elif configs.model.name == 'fc_bp2cvp_common':
        from core.models import FC_bp2cvp_common
        model = FC_bp2cvp_common(in_ch=configs.model.in_ch,
                                 out_ch=configs.model.out_ch,
                                 layer_num=configs.model.layer_num)
    elif configs.model.name == 'conv_attn_torch_bp2cvp_common_ensemble':
        from core.models import ConvAttnTorchBP2CVPCommonEnsemble
        model = ConvAttnTorchBP2CVPCommonEnsemble(
            in_ch=configs.model.in_ch,
            out_ch=configs.model.out_ch,
            kernel_size=configs.model.kernel_size,
            stride=configs.model.stride,
            padding=configs.model.padding,
            n_head=configs.model.n_head,
            conv_layer_num=configs.model.conv_layer_num,
            attn_layer_num=configs.model.attn_layer_num)
    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> nn.Module:
    if configs.criterion.name == 'mse':
        criterion = nn.MSELoss()
    elif configs.criterion.name == 'std_error_over_mean_error':
        from .criterions import std_error_over_mean_error
        criterion = std_error_over_mean_error
    elif configs.criterion.name == 'std_error_over_mean_error_abs':
        from .criterions import std_error_over_mean_error_abs
        criterion = std_error_over_mean_error_abs
    elif configs.criterion.name == 'std_error_times_mean_error_abs':
        from .criterions import std_error_times_mean_error_abs
        criterion = std_error_times_mean_error_abs
    elif configs.criterion.name == 'std_error':
        from .criterions import std_error
        criterion = std_error
    elif configs.criterion.name == 'mae':
        criterion = nn.L1Loss()
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=configs.num_epochs,
            eta_min=0)
    elif configs.scheduler.name == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1,
        )
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
