import os
import re
import json
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
import torchmetrics
import numpy as np
from torch.utils.data import DataLoader

from models import Conv2DForecasting, Conv3DForecasting, ConvLSTM


def make_data_loaders(args, dataroot):
    if args.dataset.lower() == "lyft":
        from data.lyft import LyftDataset

        eval_dataset = LyftDataset(dataroot, "test", args.p_pre, args.p_post)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")
    
    voxel_size = eval_loader.dataset.get_voxel_size()

    return eval_loader, voxel_size


def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d, exist_ok=True)


def load_ckpts(ckpt_dir, model):
    if len(os.listdir(ckpt_dir)) > 0:
        pattern = re.compile(r"model_epoch_(\d+).pth")
        epochs = []
        for f in os.listdir(ckpt_dir):
            m = pattern.findall(f)
            if len(m) > 0:
                epochs.append(int(m[0]))
        resume_epoch = max(epochs)
        ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"
        print(f"Load from checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)


def binary_soft_iou(output, label):
    inter = torch.sum(output * label)
    union = torch.sum(output + label - output * label)
    return inter / union


def eval(args):
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read config
    load_dir = args.ckpt_dir
    with open(f"{load_dir}/config.json", "r") as f:
        config = json.load(f)
    print(json.dumps(config, indent=4))
    config = Namespace(**config)

    # dataset
    eval_loader, voxel_size = make_data_loaders(config, args.dataroot)

    # model
    if config.model.lower() == "occ":
        model = Conv2DForecasting(config.p_pre+1, config.p_post+1, voxel_size[-2])
    elif config.model.lower() == "convlstm":
        model = ConvLSTM(config.p_pre+1, config.p_post+1, voxel_size[-2])
    elif config.model.lower() == "conv3d":
        model = Conv3DForecasting(config.p_pre+1, config.p_post+1)
    else:
        raise NotImplementedError(f"Model {config.model} is not supported.")
    model = model.to(device)

    # resume
    ckpt_dir = f"{load_dir}/ckpts"
    load_ckpts(ckpt_dir, model)

    # evaluation
    num_batch = len(eval_loader)
    val_metric = {"precision": 0, "recall": 0, "f1": 0, "iou": 0, "soft_iou": 0, "ap": 0}
    ious = {i: 0 for i in range(config.p_post+1)}
    soft_ious = {i: 0 for i in range(config.p_post+1)}
    with torch.no_grad():
        model.eval()
        for input, label, invalid in tqdm(eval_loader):
            input = input.to(device)
            label = label.to(device)
            invalid = invalid.to(device)
            label[label > 0] = 1

            output = model(input)
            output = torch.sigmoid(output)

            valid_mask = ~invalid
            output_all = output[valid_mask]
            label_all = label[valid_mask]
            
            precision = torchmetrics.functional.classification.binary_precision(output_all, label_all)
            recall = torchmetrics.functional.classification.binary_recall(output_all, label_all)
            f1 = 2 * precision * recall / (precision + recall)
            iou = torchmetrics.functional.classification.binary_jaccard_index(output_all, label_all)
            soft_iou = binary_soft_iou(output_all, label_all)
            ap = torchmetrics.functional.classification.binary_average_precision(output_all, label_all)

            val_metric["precision"] += precision.item()
            val_metric["recall"] += recall.item()
            val_metric["f1"] += f1.item()
            val_metric["iou"] += iou.item()
            val_metric["soft_iou"] += soft_iou.item()
            val_metric["ap"] += ap.item()

            for i in range(config.p_post+1):
                output_frame = output[:, i]
                label_frame = label[:, i]
                valid_mask = ~invalid[:, i]
                output_frame = output_frame[valid_mask]
                label_frame = label_frame[valid_mask]
                ious[i] += torchmetrics.functional.classification.binary_jaccard_index(
                    output_frame, 
                    label_frame
                ).item()
                soft_ious[i] += binary_soft_iou(output_frame, label_frame).item()
        
    for key in val_metric:
        val_metric[key] /= num_batch
    for key in ious:
        ious[key] /= num_batch
    for key in soft_ious:
        soft_ious[key] /= num_batch
    print("metrics:", json.dumps(val_metric, indent=4), end="\n")
    print("IoUs:", json.dumps(ious, indent=4))
    print("soft IoUs:", json.dumps(soft_ious, indent=4))

    with open(f"{load_dir}/metrics.json", "w") as f:
        json.dump(val_metric, f, indent=4)
    with open(f"{load_dir}/ious.json", "w") as f:
        json.dump(ious, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ckpt_dir', type=str, default='', help='Checkpoint directory')
    parser.add_argument('-r', '--dataroot', type=str, default='', help='Dataset directory')
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    eval(args)
