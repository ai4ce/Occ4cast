import os
import re
import json
import argparse
from tqdm import tqdm

import torch
import torchmetrics
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.utils.data import DataLoader

from models import Conv2DForecasting, Conv3DForecasting, ConvLSTM

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
    elif args.dataset.lower() == "argoverse":
        from data.argoverse import ArgoverseDataset

        train_loader = DataLoader(
            ArgoverseDataset(args.dataroot, "train", args.p_pre, args.p_post),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            ArgoverseDataset(args.dataroot, "valid", args.p_pre, args.p_post),
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


def resume_from_ckpts(ckpt_dir, model, optimizer, scheduler, scaler=None, amp=False):
    if len(os.listdir(ckpt_dir)) > 0:
        ckpt_path = f"{ckpt_dir}/last.pth"
        print(f"Resume training from checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if amp:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

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
    if args.model.lower() == "occ":
        model = Conv2DForecasting(args.p_pre+1, args.p_post+1, voxel_size[-2])
    elif args.model.lower() == "convlstm":
        model = ConvLSTM(args.p_pre+1, args.p_post+1, voxel_size[-2])
    elif args.model.lower() == "conv3d":
        model = Conv3DForecasting(args.p_pre+1, args.p_post+1)
    elif args.model.lower() == "conv3d_softiou":
        model = Conv3DForecasting(args.p_pre+1, args.p_post+1, use_soft_iou=True)
    else:
        raise NotImplementedError(f"Model {args.model} is not supported.")
    
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_epoch, gamma=args.lr_decay
    )

    # Scaler
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # dump config
    save_dir = f"results/{args.model}_{args.dataset}_p{args.p_pre}{args.p_post}_lr{args.lr_start}_batch{args.batch_size}{'_amp' if args.amp else ''}"
    mkdir_if_not_exists(save_dir)
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)
    print(json.dumps(args.__dict__, indent=4))

    # resume
    ckpt_dir = f"{save_dir}/ckpts"
    mkdir_if_not_exists(ckpt_dir)
    start_epoch, n_iter = resume_from_ckpts(ckpt_dir, model, optimizer, scheduler, scaler, args.amp)

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
            with autocast(enabled=args.amp):
                loss = model(input, label, invalid)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss
            pbar.set_description(f"train loss {batch_loss:.6f}")

        print(f"Epoch {epoch} train loss {train_loss / num_batch:.6f}")

        # save last model
        last_path = f"{ckpt_dir}/last.pth"
        save_last_dict = {
            "epoch": epoch,
            "n_iter": n_iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if args.amp:
            save_last_dict["scaler_state_dict"] = scaler.state_dict()
        torch.save(
            save_last_dict,
            last_path,
        )
        print(f"Last model saved to {last_path}")

        num_batch = len(val_loader)
        val_metric = {"precision": 0, "recall": 0, "f1": 0, "iou": 0, "ap": 0}
        with torch.no_grad():
            model.eval()
            for input, label, invalid in tqdm(val_loader):
                input = input.to(device)
                label = label.to(device)
                invalid = invalid.to(device)
                label[label > 0] = 1

                output = model(input, label, invalid)
                output = torch.sigmoid(output)

                valid_mask = ~invalid
                output = output[valid_mask]
                label = label[valid_mask]
                
                precision = torchmetrics.functional.classification.binary_precision(output, label)
                recall = torchmetrics.functional.classification.binary_recall(output, label)
                f1 = 2 * precision * recall / (precision + recall)
                iou = torchmetrics.functional.classification.binary_jaccard_index(output, label)
                ap = torchmetrics.functional.classification.binary_average_precision(output, label)

                val_metric["precision"] += precision.item()
                val_metric["recall"] += recall.item()
                val_metric["f1"] += f1.item()
                val_metric["iou"] += iou.item()
                val_metric["ap"] += ap.item()
            
        for key in val_metric:
            val_metric[key] /= num_batch
        print(f"Epoch {epoch} val")
        print(json.dumps(val_metric, indent=4))

        if val_metric["iou"] > best_metric:
            best_metric = val_metric["iou"]
            ckpt_path = f"{ckpt_dir}/model_epoch_{epoch}.pth"
            save_dict = {
                "epoch": epoch,
                "n_iter": n_iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            if args.amp:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(
                save_dict,
                ckpt_path,
            )
            print(f"Save model to {ckpt_path}")
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group("data")
    data_group.add_argument('-d', "--dataset", type=str, default="argoverse")
    data_group.add_argument('-r', "--dataroot", type=str, default="/home/xl3136/lyft_kitti")
    data_group.add_argument("--p_pre", type=int, default=10)
    data_group.add_argument("--p_post", type=int, default=10)

    model_group = parser.add_argument_group("model")
    model_group.add_argument('-m', "--model", type=str, default="occ")
    model_group.add_argument("--optimizer", type=str, default="Adam")  # Adam with 5e-4
    model_group.add_argument("--lr-start", type=float, default=5e-4)
    model_group.add_argument("--lr-epoch", type=float, default=5)
    model_group.add_argument("--lr-decay", type=float, default=0.1)
    model_group.add_argument("--num-epoch", type=int, default=15)
    model_group.add_argument("--batch-size", type=int, default=16)
    model_group.add_argument("--num-workers", type=int, default=4)
    model_group.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    # Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train(args)
