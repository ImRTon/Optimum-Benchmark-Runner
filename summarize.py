import argparse
import csv
import pandas as pd
import json

from pathlib import Path

from models import MODELS, IS_VISION
from utils import get_model_name

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dirs", nargs="+", help="Input directories.")
    parser.add_argument("-o", "--output_dir", default="outputs", help="Results output directory.")
    return parser.parse_args()

def main():
    args = get_args()

    input_dirs = [Path(input_dir) for input_dir in args.input_dirs]
    devices = [input_dir.parent.parent.name for input_dir in input_dirs]

    columns = ["device", "model_type", "model_name", "is_train", "bs", "dtype", "seq_len", 
               "load_memory", "load_latency", "decode_memory", "decode_throughput"]

    row = []
    for model_type, model in MODELS.items():
        for model_name, model_config in model.items():
            for bs in model_config["bs"]:
                for dtype in model_config["dtype"]:
                    for input_dir, device in zip(input_dirs, devices):
                        print(f"Processing {model_name} {dtype} bs{bs} on {device}...")
                        if model_type == "GPT" or model_type == "Bert":
                            for seq_len in model_config["seq_len"]:
                                filename = f"pytorch_{get_model_name(model_name)}_{dtype}_bs{bs}_seq{seq_len}.json"
                                data = load_json(input_dir / filename)
                                run_name = "decode" if model_type == "GPT" else "forward"
                                if "error" in data:
                                    row.append([
                                        device, 
                                        model_type, 
                                        model_name, 
                                        False, # is_train
                                        bs, 
                                        dtype, 
                                        seq_len, 
                                        None,
                                        None,
                                        None,
                                        None
                                    ])
                                else:
                                    row.append([
                                        device, 
                                        model_type, 
                                        model_name, 
                                        False, # is_train
                                        bs, 
                                        dtype, 
                                        seq_len, 
                                        data["report"]["load_model"]["memory"]["max_ram"],
                                        data["report"]["load_model"]["latency"]["mean"], 
                                        data["report"][run_name]["memory"]["max_ram"],
                                        data["report"][run_name]["throughput"]["value"]
                                    ])

                            if model_config["train"]:
                                filename = f"pytorch_{get_model_name(model_name)}_{dtype}_train_bs{bs}_seq{seq_len}.json"
                                data = load_json(input_dir / filename)

                                if "error" in data:
                                    row.append([
                                        device, 
                                        model_type, 
                                        model_name, 
                                        True, # is_train
                                        bs, 
                                        dtype, 
                                        seq_len, 
                                        None,
                                        None,
                                        None,
                                        None
                                    ])
                                else:
                                    row.append([
                                        device, 
                                        model_type, 
                                        model_name, 
                                        True, # is_train
                                        bs, 
                                        dtype, 
                                        seq_len, 
                                        data["report"]["warmup"]["memory"]["max_ram"],
                                        data["report"]["warmup"]["latency"]["mean"], 
                                        data["report"]["train"]["memory"]["max_ram"],
                                        data["report"]["train"]["throughput"]["value"]
                                    ])
                    
                        elif model_type == "ViT":
                            filename = f"pytorch_{get_model_name(model_name)}_{dtype}_bs{bs}.json"
                            data = load_json(input_dir / filename)

                            if "error" in data:
                                row.append([
                                    device, 
                                    model_type, 
                                    model_name, 
                                    False, # is_train
                                    bs, 
                                    dtype, 
                                    None, 
                                    None,
                                    None,
                                    None,
                                    None
                                ])
                            else:
                                row.append([
                                    device, 
                                    model_type, 
                                    model_name, 
                                    False, # is_train
                                    bs, 
                                    dtype, 
                                    None, 
                                    data["report"]["load_model"]["memory"]["max_ram"],
                                    data["report"]["load_model"]["latency"]["mean"], 
                                    data["report"]["forward"]["memory"]["max_ram"],
                                    data["report"]["forward"]["throughput"]["value"]
                                ])

                            if model_config["train"]:
                                filename = f"pytorch_{get_model_name(model_name)}_{dtype}_train_bs{bs}.json"
                                data = load_json(input_dir / filename)

                                if "error" in data:
                                    row.append([
                                        device, 
                                        model_type, 
                                        model_name, 
                                        True, # is_train
                                        bs, 
                                        dtype, 
                                        None, 
                                        None,
                                        None,
                                        None,
                                        None
                                    ])
                                else:
                                    row.append([
                                        device, 
                                        model_type, 
                                        model_name, 
                                        True, # is_train
                                        bs, 
                                        dtype, 
                                        None, 
                                        data["report"]["warmup"]["memory"]["max_ram"],
                                        data["report"]["warmup"]["latency"]["mean"], 
                                        data["report"]["train"]["memory"]["max_ram"],
                                        data["report"]["train"]["throughput"]["value"]
                                    ])
                        elif model_type == "Diffusion":
                            filename = f"pytorch_{get_model_name(model_name)}_{dtype}_bs{bs}.json"
                            data = load_json(input_dir / filename)

                            if "error" in data:
                                row.append([
                                    device, 
                                    model_type, 
                                    model_name, 
                                    False, # is_train
                                    bs, 
                                    dtype, 
                                    None, 
                                    None,
                                    None,
                                    None,
                                    None
                                ])
                            else:
                                row.append([
                                    device, 
                                    model_type, 
                                    model_name, 
                                    False, # is_train
                                    bs, 
                                    dtype, 
                                    None, 
                                    data["report"]["load_model"]["memory"]["max_ram"],
                                    data["report"]["load_model"]["latency"]["mean"], 
                                    data["report"]["call"]["memory"]["max_ram"],
                                    data["report"]["call"]["throughput"]["value"]
                                ])
                        else:
                            raise ValueError(f"Unknown model type: {model_type}")
                        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(row, columns=columns)
    df.to_csv(output_dir / "results.csv", index=False)

if __name__ == "__main__":
    main()