from huggingface_hub import create_repo, upload_file
create_repo("skurl/fungal-plm", repo_type="model")
upload_file(path_or_fileobj="champion.pth", path_in_repo="fungal-plm.pth", repo_id="skurl/fungal-plm")
upload_file(path_or_fileobj="MODELCARD.md", path_in_repo="README.md", repo_id="skurl/fungal-plm")