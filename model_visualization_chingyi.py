import timm
# import quant_vision_transformer
import torch
from functools import partial
import torch.nn as nn
# from timm.models.registry import register_model
from quant_vision_transformer_v3 import lowbit_VisionTransformer #, _cfg

# model = timm.models.create_model("vit_small_patch16_224", pretrained=True)
# model = timm.models.create_model("fourbits_deit_small_patch16_224", pretrained=True)
# model = timm.models.create_model("threebits_deit_small_patch16_224", pretrained=False)
model = lowbit_VisionTransformer(
        nbits=3, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=10)
# model.load_state_dict(torch.load("/Users/chingyilin/Downloads/best_checkpoint_3bit.pth", weights_only=False, map_location=torch.device('cpu'))['model'])
pth = torch.load("3bit_cifar10.pth", weights_only=False, map_location=torch.device('cpu'))
# pth = torch.load("/Users/chingyilin/Projects/Q-ViT/2bit_cifar10.pth", weights_only=False, map_location=torch.device('cpu'))
model.load_state_dict(pth['model'])
from timm.models.registry import _model_entrypoints
# import pdb; pdb.set_trace()
model_names = _model_entrypoints.keys()

print("fourbits_deit_small_patch16_224" in  model_names)
# model = timm.models.create_model("vit_small_patch16_2245", pretrained=True)

from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def build_transform(is_train, color_jitter=.4,
    aa="rand-m9-mstd0.5-inc1", train_interpolation="bicubic",
    reprob=.25, remode='pixel', recount=1, input_size=224):
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation=train_interpolation,
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

import torch
from engine import evaluate
from datasets import build_dataset
# dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
# dataset_val, _ = build_dataset(is_train=False, args=args)
import os
data_path_train = "/Volumes/PortableSSD/imagenet/ILSVRC2012_img_train_t3_bak"
root_train = os.path.join(data_path_train)
# root_train = os.path.join(data_path_train, 'train')

from pathlib import Path

def check_valid(path):
    path = Path(path)
    return not path.stem.startswith('._')

transform_train = build_transform(is_train=True)
# dataset_train = datasets.ImageFolder(root_train, transform=transform_train, is_valid_file=check_valid)
data_path_train = "/Volumes/PortableSSD/imagenet/imagenet"
# root_train = os.path.join(data_path_train)
# dataset_train = datasets.ImageNet(root=root_train, split='val')
# import pdb; pdb.set_trace()

batch_size = 32 #128
num_workers = 0
pin_mem = True
# data_loader_train = torch.utils.data.DataLoader(
#     dataset_train, # sampler=sampler_train,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     pin_memory=pin_mem,
#     drop_last=True,
# )

data_path_val = "/mnt/9e98440d-4bfc-4f8e-b75f-ab4e92d06cb1/Datasets/"
# data_path_val = "/Volumes/PortableSSD/imagenet/ILSVRC2012_img_train_t3_bak"
# data_path_val = "/Volumes/PortableSSD/imagenet/imagenet"
# root_train = os.path.join(data_path_train)
root_val = os.path.join(data_path_val)
transform_val = build_transform(is_train=False)

# transform_val = transforms.Compose([
#     # you can add other transformations in this list
#     transforms.ToTensor()
# ])

import numpy as np
# dataset_val = datasets.ImageFolder(root_val, transform=transform_val, is_valid_file=check_valid)
# dataset_val = datasets.ImageNet(root=root_val, split='val', transform=transform_val, is_valid_file=check_valid)
dataset_val = datasets.CIFAR10(root=root_val, train=False, transform=transform_val, download=True)
# dataset_val = datasets.CIFAR100(root=root_val, train=False, transform=transform_val, download=True)
dataset_val_sub = torch.utils.data.Subset(dataset_val, np.arange(100).astype(np.int32))

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, # sampler=sampler_val,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_mem,
    drop_last=False
)

device = torch.device('cpu')

# open("qkv_weightq.mem", "w").write("\n".join([str(p % 8) for p
# arr2s = lambda x: "\n".join([str(p)])
def arr2s(x, nbit=16, start=2, prefix=""):
    return "\n".join([(prefix + hex(int(p) % (2 ** nbit))[start:]) for p in x.tolist()])

def arr2single(x, nbit=16, start=2, prefix=""):
    # return " ".join("%d" % xx for xx in x)
    if nbit == 1:
        b = "".join([format(int(xx), "01b") for xx in reversed(x)])
    elif nbit == 3:
        b = "".join([format(int(xx), "03b") for xx in reversed(x)])
        # bpad = "0" * ((4 - 1) - ((len(b) + 1) % 4)) + b
        assert len(b) % 4 == 0
    elif nbit == 16:
        b = "".join([format(int(xx) % (2 ** 16), "016b") for xx in reversed(x)])
    elif nbit == 32:
        b = "".join([format(int(xx), "032b") for xx in reversed(x)])
        # import pdb; pdb.set_trace()
    # idxs = [i for i in range(0, len(b), 4)]
    tokens = [format(int(b[i:i+4], 2), "1x") for i in range(0, len(b), 4)]
    return "".join(tokens)

def arr2d2s(x, nbit=16, start=2, prefix=""):
    assert len(x.shape) == 2
    return "\n".join([arr2single(xx, nbit=nbit) for xx in x])
    # return "\n".join([(prefix + hex(int(p) % (2 ** nbit))[start:]) for p in x.tolist()])

idx_head = 1
biasq = (model.blocks[0].attn.qkv.bias / model.blocks[0].attn.qkv.act.alpha.mean() / model.blocks[0].attn.qkv.alpha).reshape(3,6,1,-1)[0, :, 0]
biask = (model.blocks[0].attn.qkv.bias / model.blocks[0].attn.qkv.act.alpha.mean() / model.blocks[0].attn.qkv.alpha).reshape(3,6,1,-1)[1, :, 0]
biasv = (model.blocks[0].attn.qkv.bias / model.blocks[0].attn.qkv.act.alpha.mean() / model.blocks[0].attn.qkv.alpha).reshape(3,6,1,-1)[2, :, 0]
prescaleq_by1024 = (model.blocks[0].attn.qkv.alpha.reshape(3, model.blocks[0].attn.num_heads,1,-1) * 1024)[0, :, 0]
prescalek_by1024 = (model.blocks[0].attn.qkv.alpha.reshape(3, model.blocks[0].attn.num_heads,1,-1) * 1024)[1, :, 0]
prescalev_by1024 = (model.blocks[0].attn.qkv.alpha.reshape(3, model.blocks[0].attn.num_heads,1,-1) * 1024)[2, :, 0]
snqq_act_bias_by256 = -(model.blocks[0].attn.norm_q.bias / model.blocks[0].attn.norm_q.weight * 256)
snqk_act_bias_by256 = -(model.blocks[0].attn.norm_k.bias / model.blocks[0].attn.norm_k.weight * 256)
stepq = (model.blocks[0].attn.q_act.alpha.mean() * 256 / model.blocks[0].attn.norm_q.weight)
stepk = (model.blocks[0].attn.k_act.alpha.mean() * 256 / model.blocks[0].attn.norm_k.weight)
    # $readmemh("../../data/snqk_scaler_head0.mem", inst_top.head1.ram_prescalek_by1024.ram);
    # $readmemh("../../data/snqkr_bias_by256_head0.mem", inst_top.head1.ram_snqk_act_bias_by256.ram);
    # $readmemh("../../data/stepkr_by256_head0.mem", inst_top.head1.ram_stepk.ram);
    # assign preboundk[i][j] = ((j - 4) * stepk[i] + (stepk[i] / 2) + snqk_act_bias_by256[i]) * ((j - 4) * stepk[i] + (stepk[i] / 2) + snqk_act_bias_by256[i]);
preboundk_sqrt = torch.outer(torch.arange(8) - 4, stepk) + .5 * stepk + snqk_act_bias_by256
preboundq_sqrt = torch.outer(torch.arange(8) - 4, stepq) + .5 * stepq + snqq_act_bias_by256
hstepv = (.5 * model.blocks[0].attn.v_act.alpha.mean() / (model.blocks[0].attn.qkv.act.alpha.mean() * model.blocks[0].attn.qkv.alpha.reshape(3, model.blocks[0].attn.num_heads, 1, -1)[2, :, 0]))
preboundv = torch.outer(torch.arange(8) * 2 - 7, hstepv.reshape(-1))
preboundvr = torch.outer(torch.arange(8) * 2 - 7, hstepv.flip(1).reshape(-1))
# wq = (model.blocks[0].attn.qkv.weight / model.blocks[0].attn.qkv.alpha.unsqueeze(1)).clamp(-4, 3).reshape(3, 6 ,-1, model.blocks[0].attn.qkv.weight.shape[1])[0, idx_head].permute(1,0)
# wk = (model.blocks[0].attn.qkv.weight / model.blocks[0].attn.qkv.alpha.unsqueeze(1)).clamp(-4, 3).reshape(3, 6 ,-1, model.blocks[0].attn.qkv.weight.shape[1])[1, idx_head].permute(1,0)
# wv = (model.blocks[0].attn.qkv.weight / model.blocks[0].attn.qkv.alpha.unsqueeze(1)).clamp(-4, 3).reshape(3, 6 ,-1, model.blocks[0].attn.qkv.weight.shape[1])[2, idx_head].permute(1,0)
wq = (model.blocks[0].attn.qkv.weight / model.blocks[0].attn.qkv.alpha.unsqueeze(1)).clamp(-4, 3).reshape(3, 6 ,-1, model.blocks[0].attn.qkv.weight.shape[1])[0].permute(0,2,1)
wk = (model.blocks[0].attn.qkv.weight / model.blocks[0].attn.qkv.alpha.unsqueeze(1)).clamp(-4, 3).reshape(3, 6 ,-1, model.blocks[0].attn.qkv.weight.shape[1])[1].permute(0,2,1)
wv = (model.blocks[0].attn.qkv.weight / model.blocks[0].attn.qkv.alpha.unsqueeze(1)).clamp(-4, 3).reshape(3, 6 ,-1, model.blocks[0].attn.qkv.weight.shape[1])[2].permute(0,2,1)
wproj = (model.blocks[0].attn.proj.weight / model.blocks[0].attn.proj.alpha.unsqueeze(1)).clamp(-4, 3).reshape(384, 6, 64).permute(1,2,0)
# wk = (model.blocks[0].attn.qkv.weight / model.blocks[0].attn.qkv.alpha.unsqueeze(1)).clamp(-4, 3).reshape(3, 6, -1, model.blocks[0].attn.qkv.weight.shape[1])[1,0].permute(1,0).round().int()[:,::-1].reshape(-1).tolist()
open("outputs/biasq_allhead.mem", "w").write(arr2s(biasq.reshape(-1).int()))
open("outputs/biasqr_allhead.mem", "w").write(arr2s(biasq.flip(1).reshape(-1).int()))
open("outputs/biask_allhead.mem", "w").write(arr2s(biask.reshape(-1).int()))
open("outputs/biaskr_allhead.mem", "w").write(arr2s(biask.flip(1).reshape(-1).int()))
open("outputs/biasv_allhead.mem", "w").write(arr2s(biasv.reshape(-1).int()))
open("outputs/biasvr_allhead.mem", "w").write(arr2s(biasv.flip(1).reshape(-1).int()))
open("outputs/snqq_scale_allhead.mem", "w").write(arr2s(prescaleq_by1024.reshape(-1).round().int()))
open("outputs/snqq_scaler_allhead.mem", "w").write(arr2s(prescaleq_by1024.flip(1).reshape(-1).round().int()))
open("outputs/snqk_scale_allhead.mem", "w").write(arr2s(prescalek_by1024.reshape(-1).round().int()))
open("outputs/snqk_scaler_allhead.mem", "w").write(arr2s(prescalek_by1024.flip(1).reshape(-1).round().int()))
open("outputs/stepq_head%d.mem" % idx_head, "w").write(arr2s(stepq.round().int()))
open("outputs/stepqr_head%d.mem" % idx_head, "w").write(arr2s(stepq.flip(0).round().int()))
open("outputs/stepk_head%d.mem" % idx_head, "w").write(arr2s(stepk.round().int()))
open("outputs/stepkr_head%d.mem" % idx_head, "w").write(arr2s(stepk.flip(0).round().int()))
open("outputs/preboundq_head%d.mem" % idx_head, "w").write(arr2d2s((preboundq_sqrt ** 2).transpose(0,1).round().int(), nbit=32))
open("outputs/preboundk_head%d.mem" % idx_head, "w").write(arr2d2s((preboundk_sqrt ** 2).transpose(0,1).round().int(), nbit=32))
open("outputs/preboundv_allhead.mem", "w").write(arr2d2s((preboundv).transpose(0,1).round().int(), nbit=16))
open("outputs/preboundvr_allhead.mem", "w").write(arr2d2s((preboundvr).transpose(0,1).round().int(), nbit=16))
open("outputs/preboundq_sign_head%d.mem" % idx_head, "w").write(arr2d2s((preboundq_sqrt < 0).int().transpose(0,1), nbit=1))
open("outputs/preboundk_sign_head%d.mem" % idx_head, "w").write(arr2d2s((preboundk_sqrt < 0).int().transpose(0,1), nbit=1))
open("outputs/snqq_bias_by256_head%d.mem" % idx_head, "w").write(arr2s(snqq_act_bias_by256.round().int(), nbit=32))
open("outputs/snqqr_bias_by256_head%d.mem" % idx_head, "w").write(arr2s(snqq_act_bias_by256.flip(0).round().int(), nbit=32))
open("outputs/snqk_bias_by256_head%d.mem" % idx_head, "w").write(arr2s(snqk_act_bias_by256.round().int(), nbit=32))
open("outputs/snqkr_bias_by256_head%d.mem" % idx_head, "w").write(arr2s(snqk_act_bias_by256.flip(0).round().int(), nbit=32))
open("outputs/stepq_by256_head%d.mem" % idx_head, "w").write(arr2s(stepq.round().int()))
open("outputs/stepqr_by256_head%d.mem" % idx_head, "w").write(arr2s(stepq.flip(0).round().int()))
open("outputs/stepk_by256_head%d.mem" % idx_head, "w").write(arr2s(stepk.round().int()))
open("outputs/stepkr_by256_head%d.mem" % idx_head, "w").write(arr2s(stepk.flip(0).round().int()))
open("outputs/hstepv_by256_allhead.mem", "w").write(arr2s(hstepv.reshape(-1).round().int()))
open("outputs/hstepvr_allhead.mem", "w").write(arr2s(hstepv.flip(1).reshape(-1).round().int()))
# open("outputs/wq1d_head%d.mem" % idx_head, "w").write(arr2d2s(wq.round().reshape(64, -1) % 8, nbit=3))
# open("outputs/wk1d_head%d.mem" % idx_head, "w").write(arr2d2s(wk.round().reshape(64, -1) % 8, nbit=3))
# open("outputs/wv1d_head%d.mem" % idx_head, "w").write(arr2d2s(wv.round().reshape(64, -1) % 8, nbit=3))
# open("outputs/wqr1d_head%d.mem" % idx_head, "w").write(arr2d2s(wq.flip(1).round().reshape(64, -1) % 8, nbit=3))
# open("outputs/wkr1d_head%d.mem" % idx_head, "w").write(arr2d2s(wk.flip(1).round().reshape(64, -1) % 8, nbit=3))
# open("outputs/wvr1d_head%d.mem" % idx_head, "w").write(arr2d2s(wv.flip(1).round().reshape(64, -1) % 8, nbit=3))

open("outputs/wq1d_allhead.mem", "w").write(arr2d2s(wq.round().reshape(6, 64, -1).reshape(6 * 64, -1) % 8, nbit=3))
open("outputs/wk1d_allhead.mem", "w").write(arr2d2s(wk.round().reshape(6, 64, -1).reshape(6 * 64, -1) % 8, nbit=3))
open("outputs/wv1d_allhead.mem", "w").write(arr2d2s(wv.round().reshape(6, 64, -1).reshape(6 * 64, -1) % 8, nbit=3))
open("outputs/wp1d_allhead.mem", "w").write(arr2d2s(wproj.round().reshape(6, 384, -1).reshape(6 * 384, -1) % 8, nbit=3))
open("outputs/wqr1d_allhead.mem", "w").write(arr2d2s(wq.flip(2).round().reshape(6, 64, -1).reshape(6 * 64, -1) % 8, nbit=3))
open("outputs/wkr1d_allhead.mem", "w").write(arr2d2s(wk.flip(2).round().reshape(6, 64, -1).reshape(6 * 64, -1) % 8, nbit=3))
open("outputs/wvr1d_allhead.mem", "w").write(arr2d2s(wv.flip(2).round().reshape(6, 64, -1).reshape(6 * 64, -1) % 8, nbit=3))
# import pdb; pdb.set_trace()
evaluate(data_loader_val, model, device)
# import pdb; pdb.set_trace()