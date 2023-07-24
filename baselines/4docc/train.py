import os
import re
import json
import argparse
from tqdm import tqdm

import torch
import torchmetrics
import numpy as np
from torch.utils.data import DataLoader

from model import OccupancyForecastingNetwork


def make_data_loaders(args):
    if args.dataset.lower() == "lyft":
        from data.lyft import LyftDataset

        train_loader = DataLoader(
            LyftDataset(args.dataroot, "train", args.p_pre, args.p_post),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            LyftDataset(args.dataroot, "valid", args.p_pre, args.p_post),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")
    
    voxel_size = train_loader.dataset.get_voxel_size()
    class_fc = val_loader.dataset.get_class_fc()

    return train_loader, val_loader, voxel_size, class_fc


def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d, exist_ok=True)


def resume_from_ckpts(ckpt_dir, model, optimizer, scheduler):
    if len(os.listdir(ckpt_dir)) > 0:
        pattern = re.compile(r"model_epoch_(\d+).pth")
        epochs = []
        for f in os.listdir(ckpt_dir):
            m = pattern.findall(f)
            if len(m) > 0:
                epochs.append(int(m[0]))
        resume_epoch = max(epochs)
        ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"
        print(f"Resume training from checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = 1 + checkpoint["epoch"]
        n_iter = checkpoint["n_iter"]
    else:
        start_epoch = 0
        n_iter = 0
    return start_epoch, n_iter


def train(args):
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset
    train_loader, val_loader, voxel_size, class_fc = make_data_loaders(args)

    # model
    model = OccupancyForecastingNetwork(
        args.p_pre+1,
        args.p_post+1,
        voxel_size[-2],
        class_fc[0],
        class_fc[1],
    )
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_epoch, gamma=args.lr_decay
    )

    # dump config
    save_dir = f"results/{args.dataset}"
    mkdir_if_not_exists(save_dir)
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # resume
    ckpt_dir = f"{save_dir}/ckpts"
    mkdir_if_not_exists(ckpt_dir)
    start_epoch, n_iter = resume_from_ckpts(ckpt_dir, model, optimizer, scheduler)

    # train
    best_metric = 0
    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        train_loss = 0
        num_batch = len(train_loader)

        for input, label, invalid in (pbar := tqdm(train_loader)):
            input = input.to(device)
            label = label.to(device)
            invalid = invalid.to(device)

            optimizer.zero_grad()
            loss = model(input, label, invalid)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss
            pbar.set_description(f"Epoch {epoch} train loss {batch_loss:.6f}")

        print(f"Epoch {epoch} train loss {train_loss / num_batch:.6f}")

        num_batch = len(val_loader)
        val_metric = {"precision": 0, "recall": 0, "f1": 0, "iou": 0, "auc": 0}
        with torch.no_grad():
            model.eval()
            for input, label, invalid in val_loader:
                input = input.to(device)
                label = label.to(device)
                invalid = invalid.to(device)

                output = model(input, label, invalid)
                output = torch.sigmoid(output)

                precision, recall = torchmetrics.functional.precision_recall(output, label)
                f1 = 2 * precision * recall / (precision + recall)
                iou = torchmetrics.functional.classification.binary_jaccard_index(output, label)
                auc = torchmetrics.functional.binary_auroc(output, label)

                val_metric["precision"] += precision.item()
                val_metric["recall"] += recall.item()
                val_metric["f1"] += f1.item()
                val_metric["iou"] += iou.item()
                val_metric["auc"] += auc.item()
            
            for key in val_metric:
                val_metric[key] /= num_batch
            print(f"Epoch {epoch} val iou {val_metric['iou']:.6f} ")

            if val_metric["iou"] > best_metric:
                best_metric = val_metric["iou"]
                ckpt_path = f"{ckpt_dir}/model_epoch_{epoch}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "n_iter": n_iter,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"Save model to {ckpt_path}")
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group("data")
    data_group.add_argument('-d', "--dataset", type=str, default="lyft")
    data_group.add_argument('-r', "--dataroot", type=str, default="/home/xl3136/lyft_kitti")
    data_group.add_argument("--p_pre", type=int, default=10)
    data_group.add_argument("--p_post", type=int, default=10)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--optimizer", type=str, default="Adam")  # Adam with 5e-4
    model_group.add_argument("--lr-start", type=float, default=5e-4)
    model_group.add_argument("--lr-epoch", type=float, default=5)
    model_group.add_argument("--lr-decay", type=float, default=0.1)
    model_group.add_argument("--num-epoch", type=int, default=15)
    model_group.add_argument("--batch-size", type=int, default=16)
    model_group.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    train(args)