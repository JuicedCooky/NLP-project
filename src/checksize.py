from huggingface_hub import HfApi, list_repo_files

api = HfApi()

files = list_repo_files(
    repo_id="HuggingFaceFW/finetranslations",
    repo_type="dataset"
)

# keep only your subset
jpn_files = [f for f in files if f.startswith("data/jpn_Jpan/")]

total_bytes = 0

for f in jpn_files:
    info = api.get_paths_info(
        repo_id="HuggingFaceFW/finetranslations",
        repo_type="dataset",
        paths=[f]
    )[0]   # returns a list, take first element

    total_bytes += info.size

print(f"Subset size: {total_bytes / (1024**3):.2f} GB")
