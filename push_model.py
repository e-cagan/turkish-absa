from huggingface_hub import HfApi

api = HfApi()
repo = "e-cagan/turkish-absa-berturk"
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path="models/absa", repo_id=repo, repo_type="model")
print("pushed:", repo)