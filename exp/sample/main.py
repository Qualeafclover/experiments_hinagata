#!/usr/bin/env python3

from pprint import pformat
from pathlib import Path
import os
import timm
import torch
import logging
import datetime
import argparse
import torchsummary
import torchvision.transforms as transforms
import shared.utils.get_dataset
from shared.utils.writer import CustomWriter
from shared.utils.config import NamedDict
from shared.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', required=True, 
        help="config file paths, later configs are prioritized")
    parser.add_argument("--save-freq", type=int, default=0, 
        help="save frequency, 0 for no saving")
    parser.add_argument("--log-freq", type=int, default=50, 
        help="log frequency during train/eval")
    parser.add_argument("--runs-dir", type=Path, default="runs", 
        help="directory to store run outputs")
    parser.add_argument("--run-name", type=str, default=datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"), 
        help="name of the run")
    parser.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu for training")
    parser.add_argument("--num-workers", type=int, default=4,
        help="Number of workers for data loading")
    args = parser.parse_args()
    return args

def create_dataset(config: NamedDict):
    train_ds, test_ds, info = getattr(shared.utils.get_dataset, config.dataset)(**config.dataset_kwargs)
    compose_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.model_input[1:]),
        # Imagenet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    train_ds = train_ds.map(lambda data: {
        "label": data["label"],
        "image": compose_fn(data["image"]),
    })

    test_ds = test_ds.map(lambda data: {
        "label": data["label"],
        "image": compose_fn(data["image"]),
    })

    return train_ds, test_ds, info

def train(model, loss_fn, dataloader, optimizer, device, log_freq):
    if str(device) == "cuda":
        device_ids = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.train()
    loss_sum = 0.
    data_processed = 0
    for i, batch in enumerate(dataloader):
        x, y = batch["image"].to(device), batch["label"].to(device)
        data_processed += len(y)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss_sum += loss.item() * len(y)
        loss.backward()
        optimizer.step()
        if log_freq > 0 and i % log_freq == 0:
            logging.info(f"Train step {i}, loss: {loss_sum/data_processed:.4f}")
    return loss_sum / data_processed, data_processed

def eval(model, loss_fn, dataloader, device, log_freq):
    if str(device) == "cuda":
        device_ids = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    loss_sum = 0.
    top1_acc = 0
    data_processed = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y = batch["image"].to(device), batch["label"].to(device)
            data_processed += len(y)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss_sum += loss.item() * len(y)
            top1_acc += (pred.argmax(dim=1) == y).sum().item()
            if log_freq > 0 and i % log_freq == 0:
                logging.info(f"Eval step {i}, loss: {loss_sum/data_processed:.4f}, top1: {top1_acc/data_processed:.4f}")
    return loss_sum / data_processed, top1_acc / data_processed

def main():
    args = parse_args()
    run_dir = args.runs_dir / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_file=run_dir / "run.log")
    logging.info(f"Arguments: \n{pformat(vars(args))}")

    config = NamedDict.from_yaml(*args.configs)
    logging.info(f"Configurations: \n{pformat(config)}")
    torch.manual_seed(config.seed)

    train_ds, eval_ds, info = create_dataset(config)
    logging.info(f"Dataset info: \n{pformat(info)}")

    if str(args.device) == "cuda":
        logging.info(f"Using CUDA devices: {list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))}")
    model = timm.create_model(config.model, pretrained=config.model_pretrained, num_classes=len(info["classes"])).to(args.device)
    logging.info(f"Model: \n{pformat(torchsummary.summary(model, input_size=config.model_input, verbose=0))}")

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    writer = CustomWriter(log_dir=run_dir, flush_secs=10)
    metrics = {"train/loss": None, "test/top1": None, "test/loss": None, "info/processed_count": 0}
    writer.add_hparams(
        hparam_dict={"model": config.model, "dataset": config.dataset, "learning_rate": config.learning_rate},
        metric_dict=metrics,
    )
    for epoch_num in range(config.train_epochs):
        logging.info(f"Epoch {epoch_num+1}/{config.train_epochs}")
        dataloader = torch.utils.data.DataLoader(
            train_ds.shuffle(100, seed=config.seed+epoch_num), 
            batch_size=config.batch_size, num_workers=args.num_workers)
        train_loss, train_processed = train(model, loss_func, dataloader, optimizer, args.device, args.log_freq)
        metrics["train/loss"] = train_loss
        metrics["info/processed_count"] += train_processed
        logging.info(f"Epoch {epoch_num+1} Train loss: {train_loss:.4f}")

        dataloader = torch.utils.data.DataLoader(
            eval_ds, batch_size=config.batch_size, num_workers=args.num_workers)
        eval_loss, eval_top1 = eval(model, loss_func, dataloader, args.device, args.log_freq)
        metrics["test/loss"] = eval_loss
        metrics["test/top1"] = eval_top1
        logging.info(f"Epoch {epoch_num+1} Eval loss: {eval_loss:.4f}, Top-1 accuracy: {eval_top1:.4f}")

        model_state = model.state_dict()
        torch.save(model_state, checkpoint_dir / "last_weight.pt")
        if args.save_freq > 0 and (epoch_num + 1) % args.save_freq == 0:
            torch.save(model_state, checkpoint_dir / f"model_epoch_{epoch_num+1}.pt")
            logging.info(f"Model saved at epoch {epoch_num+1}")
        writer.add_scalars(metrics, global_step=epoch_num+1)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
