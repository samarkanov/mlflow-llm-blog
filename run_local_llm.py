from transformers import pipeline
import torch

# Define the path to your local model directory
# Based on your `ls artifacts/` output, the model's actual files are likely inside 'artifacts/model'
LOCAL_MODEL_PATH = "artifacts/model"

# Initialize the pipeline, pointing to your local model path
# Ensure that 'device="cuda"' is only used if you have a compatible GPU and PyTorch with CUDA installed.
# If you don't have a GPU, remove 'device="cuda"' or set it to 'device="cpu"'.
# torch_dtype=torch.bfloat16 is good for compatible GPUs, otherwise consider removing it or using torch.float16/float32.
pipe = pipeline(
    "text-generation",
    model=LOCAL_MODEL_PATH,  # <--- IMPORTANT CHANGE: Use your local path here
    device="cpu", # More robust device handling
    torch_dtype=torch.bfloat16
)

# Run the pipeline with your prompt
output = pipe("Gjpfjyb vyt gj 7916", max_new_tokens=150)

# Print the generated output
print(output)

