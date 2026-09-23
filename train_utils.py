# Copyright (c) 2024 Sayed Shaun.  All rights reserved.

import os
from typing import Tuple
import wandb
import torch
from tqdm import tqdm
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup
from utils import eval_metrics_for_span_extraction, evaluation_fn, span_evaluation_fn, trainable_params


docstring = """
    Train function for training the he model for word sense disambiguation task .
    Args:

        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for training data.
        val_dataloader (torch.utils.data.DataLoader): The dataloader for validation data.
        epochs (int, optional): The number of epochs to train for. Defaults to 5.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-5.
        logging_step (int, optional): The number of steps between logging updates. Defaults to 1000.
        precision (torch.dtype, optional): The precision for training. Defaults to torch.float16.
        warmup_ratio (float, optional): The ratio of training steps for warmup. Defaults to 0.1.
        grad_clip (float, optional): The maximum norm for gradient clipping. Defaults to 1.0.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.01.
        device (str, optional): The device to train on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        output_dir (str, optional): The directory to save model checkpoints. Defaults to 'output'.
        report_to (str, optional): The reporting tool to use ('wandb' or None). Defaults to None.
"""

def add_docstring(func):
    func.__doc__ = docstring
    return func


@add_docstring
def train_fn(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-5,
        logging_step: int = 1000,
        precision: torch.dtype = torch.float16,
        warmup_ratio: float = 0.1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'output',
        report_to: str = None
    ) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(warmup_ratio * total_steps),
                    num_training_steps=total_steps)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    global_step = 1
    best_f1 = float('-inf')
    original_desc = f"Training | Params: {trainable_params(model)}"
    with tqdm(total=total_steps, desc=original_desc) as pbar:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            train_y_true, train_y_pred = [], []

            for batch in train_dataloader:
                optimizer.zero_grad()
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor)
                    else [g.to(device) for g in v]) for k, v in batch.items()
                }
                with torch.amp.autocast_mode.autocast(
                    device_type=device,
                    enabled=torch.cuda.is_available(), 
                    dtype=precision
                    ):
                    outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_y_true.extend(batch['labels'].cpu().numpy())
                train_y_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                if global_step % logging_step == 0 and global_step > 0:
                    train_accu = metrics.accuracy_score(train_y_true, train_y_pred)
                    t_p, t_r, t_f1, t_s = metrics.precision_recall_fscore_support(
                        train_y_true, train_y_pred, average='micro',zero_division=0)
                    train_y_true, train_y_pred = [], []
                    prev_desc = original_desc
                    pbar.set_description("Evaluating::")
                    v_l, v_f1, v_p, v_r, v_a = evaluation_fn(model, val_dataloader, device)
                    pbar.set_postfix({'accuracy': v_a, 'precision': v_p, 'recall': v_r, 'f1': v_f1, 'best_f1': best_f1})
                    pbar.set_description(prev_desc)
                    if report_to == 'wandb':
                        wandb.log(
                            {
                                'train/epoch': epoch,
                                'train/global_step': round(global_step, 4),
                                'train/loss': round(total_loss / logging_step, 4),
                                'train/accuracy': round(train_accu, 4),
                                'train/f1': round(t_f1, 4),
                                'train/precision': round(t_p, 4),
                                'train/recall': round(t_r, 4),
                                'validation/loss': v_l,
                                'validation/precision': v_p,
                                'validation/recall': v_r,
                                'validation/accuracy': v_a,
                                'validation/f1': v_f1,
                                'validation/best_f1': best_f1
                            }
                        )
                    total_loss = 0.0

                    # Save checkpoints
                    if v_f1 > best_f1:
                        best_f1 = v_f1
                        torch.save(model.state_dict(), os.path.join(output_dir, f"step-{global_step}-f1-{v_f1}.pt"))
                        # Sorted by F1 score and keep 3 checkpoints
                        checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.pt') and '-' in f]
                        checkpoints = sorted(
                            checkpoint_files,
                            key=lambda x: float(x.split('-')[3].replace(".pt", "")),
                            reverse=True
                        )
                        for ckpt in checkpoints[3:]:
                            os.remove(os.path.join(output_dir, ckpt))

                global_step += 1
                pbar.update(1)


@add_docstring
def span_train_fn(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-5,
        logging_step: int = 1000,
        precision: torch.dtype = torch.float16,
        warmup_ratio: float = 0.1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'output',
        report_to: str = None
    ) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(warmup_ratio * total_steps),
                    num_training_steps=total_steps)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    global_step = 0
    best_f1 = float('-inf')
    original_desc = f"Training | Params: {trainable_params(model)}"
    with tqdm(total=total_steps, desc=original_desc) as pbar:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            train_start_true, train_start_pred = [], []
            train_end_true, train_end_pred = [], []

            for batch in train_dataloader:
                optimizer.zero_grad()
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor)
                    else [g.to(device) for g in v]) for k, v in batch.items()
                }
                with torch.amp.autocast_mode.autocast(
                    device_type=device,
                    enabled=torch.cuda.is_available(), 
                    dtype=precision
                    ):
                    outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_start_true.extend(batch['start_positions'].cpu().numpy())
                train_start_pred.extend(outputs.start_logits.argmax(dim=1).cpu().numpy())
                train_end_true.extend(batch['end_positions'].cpu().numpy())
                train_end_pred.extend(outputs.end_logits.argmax(dim=1).cpu().numpy())
                # Logging
                if global_step % logging_step == 0 and global_step > 0:
                    start_f1, end_f1, joint_f1, em = eval_metrics_for_span_extraction(
                        pred_start=train_start_pred, 
                        pred_end=train_end_pred, 
                        start_positions=train_start_true, 
                        end_positions=train_end_true
                    )
                    train_start_true, train_start_pred = [], []
                    train_end_true, train_end_pred = [], []
                    prev_desc = original_desc
                    pbar.set_description("Evaluating::")
                    val_loss, v_start_f1, v_end_f1, v_joint_f1, v_em = span_evaluation_fn(model, val_dataloader, device)
                    pbar.set_postfix({'start_f1': v_start_f1, 'end_f1': v_end_f1, 'joint_f1': v_joint_f1, 'em': v_em})
                    pbar.set_description(prev_desc)
                    if report_to == 'wandb':
                        wandb.log(
                            {
                                'train/epoch': epoch,
                                'train/global_step': global_step,
                                'train/loss': round(total_loss / logging_step, 4),
                                'train/start_f1': start_f1,
                                'train/end_f1': end_f1,
                                'train/joint_f1': joint_f1,
                                'train/exact_match': em,
                                'validation/loss': val_loss,
                                'validation/start_f1': v_start_f1,
                                'validation/end_f1': v_end_f1,
                                'validation/joint_f1': v_joint_f1,
                                'validation/exact_match': v_em,
                                'validation/best_f1': best_f1
                            }
                        )
                    total_loss = 0.0

                    # Save checkpoints
                    if v_joint_f1 > best_f1:
                        best_f1 = v_joint_f1
                        torch.save(model.state_dict(), os.path.join(output_dir, f"step-{global_step}-f1-{best_f1}.pt"))
                        # Sorted by F1 score and keep 3 checkpoints
                        checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.pt') and '-' in f]
                        checkpoints = sorted(
                            checkpoint_files,
                            key=lambda x: float(x.split('-')[3].replace(".pt", "")),
                            reverse=True
                        )
                        for ckpt in checkpoints[3:]:
                            os.remove(os.path.join(output_dir, ckpt))

                global_step += 1
                pbar.update(1)