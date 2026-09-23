# Copyright (c) 2025, Sayed Shaun.  All rights reserved.

import os
import json
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Union
from sklearn import metrics
from transformers import AutoTokenizer


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the tokenizer for the specified model.
    Args:
        model_name (str): The name of the model to load the tokenizer for.
    Returns:
        AutoTokenizer: The tokenizer for the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<classify>", "</classify>"]}
    )
    return tokenizer


def trainable_params(model: torch.nn.Module) -> Union[str, float]:
    """
    Calculate the number of trainable parameters in the model.
    Args:
        model (torch.nn.Module): The model to analyze.
    Returns:
        str: A string representation of the number of trainable parameters in millions.
    """
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            params += param.numel()
    return f"{params / 1e6:.2f}M"


def seed_everything(seed: Union[int, None] = None) -> None:
    """
    Set random seed for reproducibility.
    Args:
        seed (int, optional): The seed value. If None, no seed is set.
    Returns:
        None
    """
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_model_weights(model: torch.nn.Module, weight_dir: str, device: str) -> torch.nn.Module:
    """
    Load model weights from the specified directory. 
    Args:
        model (torch.nn.Module): The model to load the weights into.
        weight_dir (str): The directory containing the model weights.
        device (str): The device to load the model onto ('cpu' or 'cuda').
    Returns:
        torch.nn.Module: The model with loaded weights.
    Raises:
        FileNotFoundError: If the weight directory does not exist or contains no valid weights.
    Raises:
        ValueError: If the weight directory is empty or contains no valid model weights.
    """
    if not os.path.exists(weight_dir):
        raise FileNotFoundError(f"No weights found in {weight_dir}")
    
    weight_list = [f for f in os.listdir(weight_dir) if f.endswith((".pt", ".bin", ".ckpt"))]
    if not weight_list:
        raise FileNotFoundError(f"No valid model weights (.pt/.bin/.ckpt) found in {weight_dir}")
    
    path = sorted(weight_list, key=lambda x: x.split("-")[-1])[-1]
    full_path = os.path.join(weight_dir, path)
    print("Loading model from:", full_path)
    state_dict = torch.load(full_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model



def eval_metrics_for_span_extraction(
        pred_start: List[int], pred_end: List[int], 
        start_positions: List[int], end_positions: List[int]
        ) -> Tuple[float, float, float, float]:
    """
    Calculate evaluation F1 metrics for span extraction tasks.
    Args:
        pred_start (List[int]): Predicted start positions.
        pred_end (List[int]): Predicted end positions.
        start_positions (List[int]): True start positions.
        end_positions (List[int]): True end positions.
    Returns:
        Tuple[float, float, float]: A tuple containing the start F1 score, 
        end F1 score, and joint F1 score.
    Raises:
        ValueError: If the lengths of the input lists do not match.
    """
    if not (len(pred_start) == len(pred_end) == len(start_positions) == len(end_positions)):
        raise ValueError("All input lists must have the same length.")
    
    start_f1 = metrics.f1_score(
        start_positions, pred_start, average='micro',
        zero_division=0
    )
    end_f1 = metrics.f1_score(    
        end_positions, pred_end, average='micro',
        zero_division=0
    )
    exact_match = [
        (pred_start[i] == start_positions[i] and
         pred_end[i] == end_positions[i])
        for i in range(len(pred_start))
    ]
    em = sum(exact_match) / len(exact_match)
    joint_f1 = np.mean([start_f1, end_f1, em])
    return start_f1, end_f1, joint_f1, em


def evaluation_fn(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
        device: str, disable_tqdm_bar: bool = True
        ) -> Tuple[float, float, float, float, float]:
    """Evaluate the model on the given dataloader.
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for evaluation.
        device (str): The device to run the evaluation on ('cpu' or 'cuda').
    Returns:
        Tuple[float, float, float, float, float]: A tuple containing the loss, 
        F1 score, precision, recall, and accuracy.
    """
    with torch.no_grad():
        y_true, y_pred = [], []
        model.eval()
        total_loss = 0.0
        for batch in tqdm(dataloader, disable=disable_tqdm_bar):
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor)
                else [g.to(device) for g in v]) for k, v in batch.items()
            }
            outputs = model(**batch)  # (batch, n_sense)
            y_true.extend(batch['labels'].cpu().numpy())
            y_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            loss = outputs.loss.item()
            total_loss += loss

        loss = total_loss / len(dataloader)
        precision, recall, f1, support = metrics.precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0)
        acc = metrics.accuracy_score(y_true, y_pred)
        y_true, y_pred = [], []
        model.train()

    return round(loss,4), round(f1,4), round(precision,4), round(recall,4), round(acc,4)


def span_evaluation_fn(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
        device: str,  disable_tqdm_bar: bool = True
        ) -> Tuple[float, float, float, float, float]:
    """Evaluate the model on the given dataloader for span extraction tasks.
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for evaluation.
        device (str): The device to run the evaluation on ('cpu' or 'cuda').
    Returns:
        Tuple[float, float, float, float, float]: A tuple containing the loss, 
        start F1 score, end F1 score, joint F1 score, and accuracy.
    """
    with torch.no_grad():
        start_true, start_pred = [], []
        end_true, end_pred = [], []
        model.eval()
        total_loss = 0.0
        for batch in tqdm(dataloader, disable=disable_tqdm_bar):
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor)
                else [g.to(device) for g in v]) for k, v in batch.items()
            }
            outputs = model(**batch)  # (batch, n_sense)
            start_true.extend(batch['start_positions'].cpu().numpy())
            start_pred.extend(outputs.start_logits.argmax(dim=1).cpu().numpy())
            end_true.extend(batch['end_positions'].cpu().numpy())
            end_pred.extend(outputs.end_logits.argmax(dim=1).cpu().numpy())
            total_loss += outputs.loss.item()

        loss = total_loss / len(dataloader)
        start_f1, end_f1, joint_f1, em = eval_metrics_for_span_extraction(
            pred_start=start_pred, 
            pred_end=end_pred, 
            start_positions=start_true, 
            end_positions=end_true
        )
        start_true, start_pred = [], []
        end_true, end_pred = [], []
        model.train()
        return round(loss, 4), round(start_f1, 4), round(end_f1, 4), round(joint_f1, 4), round(em, 4)
    

def plot_metrics(json_path: str, save_path: str = None) -> None:
    """
    Creates a single line plot containing all numeric metrics across datasets.
    Args:
        json_path (str): Path to the metrics.json file.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    assert save_path is not None, "Please provide a valid save path for the plots."
    def style_fn():
        sns.set_style(style="darkgrid")
        plt.figure(figsize=(10, 4))

    style_fn()
    df = pd.read_json(json_path)
    numeric_df = df.drop(index=["pos", "architecture"], errors="ignore")
    df_t = numeric_df.transpose()

    sns.lineplot(data=df_t, dashes=False, markers=True)
    plt.title("Metrics Across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.1)  # Reduce padding
    plt.savefig(f"{save_path}/line.png", bbox_inches='tight')
    plt.close()

    # Heatmap
    style_fn()
    sns.heatmap(df_t.apply(pd.to_numeric, errors='coerce').transpose(), 
                annot=True, cmap="YlGnBu", cbar_kws={'label': 'Value'})
    plt.title("Heatmap of Metrics across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Metric")
    plt.tight_layout(pad=0.1)
    plt.savefig(f"{save_path}/heatmap.png", bbox_inches='tight')
    plt.close()
