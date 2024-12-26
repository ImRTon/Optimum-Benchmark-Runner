# Optimum Benchmark Runner
A simple benchmark runner for the [Optimum-benchmark](https://github.com/huggingface/optimum-benchmark) project.

## Installation
```bash
pip install diffusers[torch]
pip install transformers
pip install huggingface_hub
pip install git+https://github.com/huggingface/optimum-benchmark.git
```

## Preparing the benchmark
Login to the huggingface-cli in order to be able to access some limited access models.  
```python
huggingface-cli login
```

Make sure you have the following models accessible in your huggingface account:
* https://huggingface.co/meta-llama/Llama-3.2-3B
* https://huggingface.co/meta-llama/Llama-3.1-8B

## Running the benchmark
```bash
python main.py
```