"""Sync local data/ with the shared HuggingFace dataset repo.

Initializes data/ as a git repo pointing at the HF dataset (first run only),
then uses git pull/push for incremental syncs. Since data/ already contains
most files, only the delta is transferred.

Usage:
    uv run python scripts/utils/sync_HF_data.py          # pull then push
    uv run python scripts/utils/sync_HF_data.py --pull    # pull only
    uv run python scripts/utils/sync_HF_data.py --push    # push only
    uv run python scripts/utils/sync_HF_data.py --init    # one-time setup
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "SGTR-Geodesic/self-rec-results"
LOCAL_DIR = Path("data")


def _get_hf_git_url():
    """Build the authenticated git URL for the HF dataset repo."""
    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    if not token:
        print("Error: HF_TOKEN not set and no cached token found.")
        sys.exit(1)
    return f"https://user:{token}@huggingface.co/datasets/{REPO_ID}"


def _run_git(*args, **kwargs):
    """Run a git command inside data/."""
    return subprocess.run(
        ["git", "-C", str(LOCAL_DIR)] + list(args),
        check=True,
        **kwargs,
    )


def _is_initialized():
    """Check if data/ is already a git repo with the HF remote."""
    return (LOCAL_DIR / ".git").exists()


def init():
    """One-time setup: make data/ track the HF repo without re-downloading."""
    if _is_initialized():
        print("Already initialized — data/ is a git repo.")
        return

    url = _get_hf_git_url()

    print("Installing git-lfs...")
    subprocess.run(["git", "lfs", "install", "--skip-repo"], check=True)

    print(f"Initializing data/ as git repo tracking {REPO_ID}...")
    _run_git("init")
    _run_git("remote", "add", "origin", url)
    _run_git("lfs", "install", "--local")

    # Fetch remote history without downloading file contents
    print("Fetching remote refs (metadata only)...")
    _run_git("fetch", "origin")

    # Reset to match remote HEAD — marks existing local files as tracked
    # without overwriting them. Only truly new/changed files get downloaded.
    print("Aligning local state with remote (no file downloads for matching files)...")
    _run_git("reset", "origin/main")

    print("✓ Initialized. Run --pull to download any new files from your colleague.\n")


def pull():
    """Pull new/changed files from HF. Stashes local changes, pulls, then pops."""
    if not _is_initialized():
        print("data/ not initialized. Run with --init first.")
        sys.exit(1)

    print(f"Pulling from {REPO_ID}...")

    # Stash any local changes (staged or unstaged)
    _run_git("add", "-A")
    stash_result = subprocess.run(
        ["git", "-C", str(LOCAL_DIR), "stash", "--include-untracked"],
        capture_output=True, text=True,
    )
    stashed = "No local changes" not in stash_result.stdout

    _run_git("pull", "origin", "main", "--rebase")

    if stashed:
        _run_git("stash", "pop")

    print("✓ Pull complete\n")


def push():
    """Stage all changes in data/ (except analysis/) and push to HF.
    Always pulls first to avoid overwriting remote changes."""
    if not _is_initialized():
        print("data/ not initialized. Run with --init first.")
        sys.exit(1)

    # Ensure .gitignore excludes analysis/ and figures/
    gitignore = LOCAL_DIR / ".gitignore"
    ignore_entries = ["analysis/", "figures/"]
    existing = gitignore.read_text() if gitignore.exists() else ""
    with open(gitignore, "a") as f:
        for entry in ignore_entries:
            if entry not in existing:
                f.write(f"\n{entry}\n")

    print(f"Staging and pushing to {REPO_ID}...")
    _run_git("add", "-A")

    # Check if there's anything to commit
    result = subprocess.run(
        ["git", "-C", str(LOCAL_DIR), "diff", "--cached", "--quiet"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("Nothing new to push.\n")
        return

    _run_git("commit", "-m", "Update data")

    # Fetch + rebase before pushing to avoid overwriting remote changes
    print("Fetching remote to check for new commits...")
    _run_git("fetch", "origin")
    _run_git("rebase", "origin/main")

    _run_git("push", "origin", "HEAD:main")
    print("✓ Push complete\n")


def main():
    parser = argparse.ArgumentParser(description="Sync data/ with HuggingFace")
    parser.add_argument("--init", action="store_true",
                        help="One-time setup: initialize data/ as HF git repo")
    parser.add_argument("--pull", action="store_true", help="Pull only")
    parser.add_argument("--push", action="store_true", help="Push only")
    args = parser.parse_args()

    if args.init:
        init()
        return

    if not args.pull and not args.push:
        pull()
        push()
    else:
        if args.pull:
            pull()
        if args.push:
            push()


if __name__ == "__main__":
    main()
