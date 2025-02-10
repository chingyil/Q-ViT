import timm
# import quant_vision_transformer
import torch
from functools import partial
import torch.nn as nn
# from timm.models.registry import register_model
from quant_vision_transformer import lowbit_VisionTransformer #, _cfg

# model = timm.models.create_model("vit_small_patch16_224", pretrained=True)
# model = timm.models.create_model("fourbits_deit_small_patch16_224", pretrained=True)
# model = timm.models.create_model("threebits_deit_small_patch16_224", pretrained=False)
model = lowbit_VisionTransformer(
        nbits=3, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
model.load_state_dict(torch.load("/Users/chingyilin/Downloads/best_checkpoint_3bit.pth", weights_only=False, map_location=torch.device('cpu'))['model'])
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

batch_size = 8
num_workers = 0
pin_mem = True
# data_loader_train = torch.utils.data.DataLoader(
#     dataset_train, # sampler=sampler_train,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     pin_memory=pin_mem,
#     drop_last=True,
# )

# data_path_val = "/Volumes/PortableSSD/imagenet/ILSVRC2012_img_train_t3_bak"
data_path_val = "/Volumes/PortableSSD/imagenet/imagenet"
# root_train = os.path.join(data_path_train)
root_val = os.path.join(data_path_val)
transform_val = build_transform(is_train=False)

# transform_val = transforms.Compose([
#     # you can add other transformations in this list
#     transforms.ToTensor()
# ])

import numpy as np
# dataset_val = datasets.ImageFolder(root_val, transform=transform_val, is_valid_file=check_valid)
dataset_val = datasets.ImageNet(root=root_val, split='val', transform=transform_val, is_valid_file=check_valid)
dataset_val_sub = torch.utils.data.Subset(dataset_val, np.arange(1000).astype(np.int32))

data_loader_val = torch.utils.data.DataLoader(
    dataset_val_sub, # sampler=sampler_val,
    batch_size=int(1.5 * batch_size),
    num_workers=num_workers,
    pin_memory=pin_mem,
    drop_last=False
)

device = torch.device('cpu')
import pdb; pdb.set_trace()
evaluate(data_loader_val, model, device)
import pdb; pdb.set_trace()