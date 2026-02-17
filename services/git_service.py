import os
import shutil
from git import Repo

def clone_repo(repo_link: str, dest_dir: str) -> str:
    """
    Clone a repo into dest_dir. If dest exists, delete it first.
    Returns the local path.
    """
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
    Repo.clone_from(repo_link, dest_dir)
    return dest_dir
