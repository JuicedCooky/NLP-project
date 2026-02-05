from huggingface_hub import snapshot_download

from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="HuggingFaceFW/finetranslations",
    repo_type="dataset"
)

# keep only your subset folder
jpn_files = [f for f in files if f.startswith("data/jpn_Jpan/")]

print("Total files in subset:", len(jpn_files))

half = len(jpn_files) // 4
first_half = jpn_files[:half]

print("Downloading only:", len(first_half), "files")

folder = snapshot_download(
                "HuggingFaceFW/finetranslations", 
                repo_type="dataset",
                local_dir="./finetranslations/",
                # download the Czech filtered data
                allow_patterns=first_half,
                )

print("Downloaded to:", folder)