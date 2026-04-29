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

    # Ensure files matching LFS patterns are stored via LFS.
    # HuggingFace rejects pushes with any non-LFS file >10MiB, so scan the
    # entire working tree for large files and promote any that aren't
    # currently LFS-tracked. `git lfs ls-files` shows files LFS already
    # handles (either via .gitattributes rules or a prior explicit import).
    LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10 MiB, HF's hard cap

    lfs_result = subprocess.run(
        ["git", "-C", str(LOCAL_DIR), "lfs", "ls-files", "--name-only"],
        capture_output=True, text=True,
    )
    lfs_tracked = set(lfs_result.stdout.strip().splitlines()) if lfs_result.returncode == 0 else set()

    promoted: list[Path] = []
    for path in LOCAL_DIR.rglob("*"):
        if not path.is_file():
            continue
        # Skip anything inside .git
        if ".git" in path.parts:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= LARGE_FILE_THRESHOLD:
            continue
        rel = str(path.relative_to(LOCAL_DIR))
        if rel in lfs_tracked:
            continue
        print(f"  Re-staging {rel} for LFS ({size // (1024 * 1024)} MiB)")
        # Remove from git's index so `git add` re-applies LFS filter rules
        # on the next add. Falls through silently if not currently tracked.
        subprocess.run(
            ["git", "-C", str(LOCAL_DIR), "rm", "--cached", rel],
            capture_output=True,
        )
        promoted.append(path)

    # If any files were promoted, make sure their extensions are covered by
    # .gitattributes. Without the filter rule, `git add` won't route them
    # through LFS even after `rm --cached`.
    if promoted:
        attrs_path = LOCAL_DIR / ".gitattributes"
        attrs = attrs_path.read_text() if attrs_path.exists() else ""
        exts_needed = {p.suffix for p in promoted if p.suffix}
        added_attrs = []
        for ext in sorted(exts_needed):
            pattern = f"*{ext}"
            line = f"{pattern} filter=lfs diff=lfs merge=lfs -text"
            if line not in attrs:
                attrs += ("\n" if attrs and not attrs.endswith("\n") else "") + line + "\n"
                added_attrs.append(line)
        if added_attrs:
            attrs_path.write_text(attrs)
            print(f"  Added {len(added_attrs)} LFS filter rules to .gitattributes")

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
