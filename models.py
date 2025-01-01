MODELS = {
    "GPT":
        {
            "meta-llama/Llama-3.2-3B": {
                "bs": [1],
                "seq_len": [128, 512, 2048, 4096, 8192],
                "train": False,
                "dtype": ["float16", "bfloat16", "float32"]
            }, "meta-llama/Llama-3.1-8B": {
                "bs": [1],
                "seq_len": [128, 512, 2048, 4096, 8192],
                "train": False,
                "dtype": ["float16", "bfloat16", "float32"]
            }
        },
    "Bert":
        {
            "google-bert/bert-base-uncased": {
                "bs": [1, 2, 4, 8, 16],
                "seq_len": [128, 256, 512],
                "train": True,
                "dtype": ["float16", "bfloat16", "float32"]
            }
        },
    "ViT":
        {
            "google/vit-base-patch16-224": {
                "bs": [1, 2, 4, 8, 16],
                "train": True,
                "dtype": ["float32"]
            }, "facebook/dinov2-large": {
                "bs": [1, 2, 4, 8, 16],
                "train": False,
                "dtype": ["float32"]
            },
        },
    "Diffusion": 
        {
            "stable-diffusion-v1-5/stable-diffusion-v1-5": {
                "bs": [1],
                "train": False,
                "dtype": ["float16", "bfloat16"]
            }, "stabilityai/stable-diffusion-xl-base-1.0": {
                "bs": [1],
                "train": False,
                "dtype": ["float16", "bfloat16"]
            }, "black-forest-labs/FLUX.1-schnell": {
                "bs": [1],
                "train": False,
                "dtype": ["float16", "bfloat16"]
            }
        }
}

IS_VISION = {
    "GPT": False,
    "Bert": False,
    "ViT": True,
    "ResNet": True,
    "Diffusion": True
}