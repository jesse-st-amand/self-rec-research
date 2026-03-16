from huggingface_hub import HfApi, create_repo

REPO_ID = "SGTR-Geodesic/self-rec-results"

api = HfApi()
# Create a private dataset repo (run once)
create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    private=True,
    exist_ok=True,
)

# Upload dataset card (README.md at repo root)
api.upload_file(
    path_or_fileobj="data/results/README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)

# --- Upload data/ folder (resumable, multi-threaded, resilient to errors) ---
api.upload_large_folder(
    folder_path="data",
    repo_id=REPO_ID,
    repo_type="dataset",
    ignore_patterns=["analysis/*"],
)
