import os
import json

from pathlib import Path
from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, TrainingConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

from utils import simplify_exception, get_model_name
from models import MODELS, IS_VISION

PUSH_REPO_ID = os.environ.get("PUSH_REPO_ID", None)

def run_benchmark(scenario_config, launcher_config, backend_config, name, output_dir="outputs"):
    if Path(f"{output_dir}/{name}.json").exists():
        print(f"Existed result, skipping {name}.")
        return
    try:
        benchmark_config = BenchmarkConfig(
            name=name,
            launcher=launcher_config,
            scenario=scenario_config,
            backend=backend_config,
            print_report=True,
            log_report=True,
        )
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
        benchmark.save_json(f"outputs/{name}.json")
    except Exception as e:
        error = {"error": simplify_exception(str(e))}
        with open(f"{output_dir}/{name}.json", 'w') as f:
            json.dump(error, f)

if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", "INFO")
    to_file = os.environ.get("LOG_TO_FILE", "0") == "1"
    setup_logging(level=level, to_file=to_file, prefix="MAIN-PROCESS")

    launcher_config = ProcessConfig(device_isolation=True, device_isolation_action="warn")

    for model_type, models in MODELS.items():
        for model_name, model_config in models.items():
            BENCHMARK_NAME = f"pytorch_{get_model_name(model_name)}"
            MODEL = model_name

            for bs in model_config["bs"]:
                for dtype in model_config["dtype"]:
                    backend_config = PyTorchConfig(
                        device="cuda",
                        device_ids="0", 
                        no_weights=True if model_type != "Diffusion" else False, 
                        model=MODEL, 
                        torch_dtype=dtype
                    )

                    if IS_VISION[model_type]:
                        infer_scenario_config = InferenceConfig(memory=True, latency=True, input_shapes={"batch_size": bs})
                        run_benchmark(infer_scenario_config, launcher_config, backend_config, f"{BENCHMARK_NAME}_{dtype}_bs{bs}")

                    else:
                        for seq_len in model_config["seq_len"]:
                            infer_scenario_config = InferenceConfig(memory=True, latency=True, input_shapes={"batch_size": bs, "sequence_length": seq_len})
                            run_benchmark(infer_scenario_config, launcher_config, backend_config, f"{BENCHMARK_NAME}_{dtype}_bs{bs}_seq{seq_len}")

                    if model_config["train"]:
                        if IS_VISION[model_type]:
                            train_scenario_config = TrainingConfig(memory=True, latency=True, training_arguments={"per_device_train_batch_size": bs})
                            run_benchmark(train_scenario_config, launcher_config, backend_config, f"{BENCHMARK_NAME}_{dtype}_train_bs{bs}")
                        else:
                            for seq_len in model_config["seq_len"]:
                                train_scenario_config = TrainingConfig(
                                    memory=True, latency=True, 
                                    dataset_shapes={"sequence_length": seq_len},
                                    training_arguments={"per_device_train_batch_size": bs}
                                )
                                run_benchmark(train_scenario_config, launcher_config, backend_config, f"{BENCHMARK_NAME}_{dtype}_train_bs{bs}_seq{seq_len}")
