import os
import json
import torch
import argparse
from dataset import WSDDataset, SpanDataset
from data_builder import DataBuilder
from model import WSDModel, SpanExtractionModel
from utils import get_tokenizer, load_model_weights, seed_everything
from torch.utils.data import DataLoader
from train_utils import evaluation_fn, span_evaluation_fn


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args.model_name)
    if args.architecture == "cosine":
        model = WSDModel(args.model_name, tokenizer=tokenizer, n_sense=args.num_sense).to(device)
        model = load_model_weights(model, args.weight_dir, device)
        ds = WSDDataset(
            dataframe=DataBuilder(args.data_dir, args.pos, args.seed).to_pandas(), 
            tokenizer=tokenizer, 
            n_sense=args.num_sense, 
            max_seq_length=args.max_length,
        )
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loss, f1, precision, recall, acc = evaluation_fn(model, dataloader, device, False)
        report = {
            "loss": loss,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": acc,
            "pos": args.pos,
            "architecture": args.architecture,
        }
        print(report)
        output_filename = f"{args.data_dir.split('/')[-1]}.json"
        with open(os.path.join(args.output_dir, output_filename), "w") as f:
            json.dump(report, f, indent=4)
    else:
        model = SpanExtractionModel(args.model_name, tokenizer=tokenizer, n_sense=args.num_sense).to(device)
        model = load_model_weights(model, args.weight_dir, device)
        ds = SpanDataset(
            dataframe=DataBuilder(args.data_dir, args.pos, args.seed).to_pandas(), 
            tokenizer=tokenizer, 
            max_seq_length=args.max_length
        )
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loss, start_f1, end_f1, joint_f1, em = span_evaluation_fn(model, dataloader, device, False)
        report = {
            "loss": loss,
            "start_f1": start_f1,
            "end_f1": end_f1,
            "exact_match": em,
            "joint_f1": joint_f1,
            "pos": args.pos,
            "architecture": args.architecture,
        }
        print(report)
        output_filename = f"{args.data_dir.split('/')[-1]}.json"
        with open(os.path.join(args.output_dir, output_filename), "w") as f:
            json.dump(report, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--weight_dir", type=str, required=True, help="Path to model weight directory")
    parser.add_argument("--pos", type=str, default="ALL", help="POS tag to use")
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducibility")
    parser.add_argument("--num_sense", type=int, default=5, help="Number of sense")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of input")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--architecture", required=True, type=str, help="Model architecture to use")
    args = parser.parse_args()
    main(args)