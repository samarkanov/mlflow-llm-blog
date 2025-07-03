from huggingface_hub import snapshot_download

# This will download all the necessary files into the './artifacts/model' directory
# Ensure this matches the path you use in your pipeline
print("Downloading model to artifacts/model...")
snapshot_download(repo_id="google/gemma-3-1b-pt", local_dir="./artifacts/model")
print("Download complete.")
