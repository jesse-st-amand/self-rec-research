"""Generic RunPod job dispatch for alpaca_eval pipeline.

Reuses SGTR-RL's RunPodClient for pod lifecycle management.
Builds custom startup scripts that run arbitrary commands on a GPU pod
with the full self-rec-research workspace available.

The pod:
1. Clones self-rec-research + editable deps (same as training pods)
2. Installs dependencies via uv sync
3. Runs the specified command
4. Writes results to the network volume
5. Exits (and is auto-deleted if terminate_on_exit=True)

Results are retrieved from the network volume path, which is shared
between the pod and the local machine.
"""

import os
import shlex
from datetime import UTC, datetime
from pathlib import Path

import yaml

from sgtr_rl.scripts.runpod_utils import (
    RunPodClient,
    infer_repo_ref,
    infer_repo_url,
    resolve_runpod_env,
)
from sgtr_rl.runtime_config import load_runtime_config

# Default location for tier-based GPU configs
GPU_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "gpu_configs"


def get_runtime_for_model(
    model_name: str,
    dispatch_type: str = "runpod",
) -> str | None:
    """Resolve the runtime YAML path for a model based on its GPU tier.

    Looks up the model's tier from MODEL_GPU_TIER and returns the path
    to gpu_configs/{dispatch_type}_{tier}.yaml.

    Returns None if the model is not an hf/ model.
    """
    from self_rec_framework.src.helpers.model_names import get_gpu_tier

    tier = get_gpu_tier(model_name)
    if tier is None:
        return None

    runtime_path = GPU_CONFIGS_DIR / f"{dispatch_type}_{tier}.yaml"
    if runtime_path.exists():
        return str(runtime_path)

    # Fall back to medium if specific tier not found
    fallback = GPU_CONFIGS_DIR / f"{dispatch_type}_medium.yaml"
    if fallback.exists():
        print(f"  ⚠ No {dispatch_type}_{tier}.yaml found, falling back to medium")
        return str(fallback)

    return None


def _build_startup_script(
    *,
    repo_url: str,
    repo_ref: str,
    editable_deps: list,
    workspace_subdir: str,
    cache_dir: str | None,
    command: str,
) -> str:
    """Build a pod startup script that clones the workspace and runs a command."""
    lines = [
        "set -euo pipefail",
        "python -m pip install --upgrade pip uv",
        "uv python install 3.12",
        "mkdir -p /root/workspace",
        "cd /root/workspace",
        # Clone or pull if already exists (pod restarts reuse the same filesystem)
        f"if [ -d {shlex.quote(workspace_subdir)} ]; then "
        f"cd {shlex.quote(workspace_subdir)} && git fetch && git checkout {shlex.quote(repo_ref)} && git pull origin {shlex.quote(repo_ref)} || true; "
        f"else "
        f"git clone {shlex.quote(repo_url)} {shlex.quote(workspace_subdir)} && "
        f"cd {shlex.quote(workspace_subdir)} && "
        f"git checkout {shlex.quote(repo_ref)}; "
        f"fi",
    ]

    # Clone editable dependencies (skip if already cloned)
    for dep in editable_deps:
        parent = str(Path(dep.path).parent)
        lines.append(f"mkdir -p {shlex.quote(parent)}")
        lines.append(f"if [ ! -d {shlex.quote(dep.path)} ]; then git clone {shlex.quote(dep.repo_url)} {shlex.quote(dep.path)}; fi")
        if dep.ref:
            lines.append(
                f"git -C {shlex.quote(dep.path)} checkout {shlex.quote(dep.ref)}"
            )

    # Set up HF cache — use network volume if available, otherwise container disk
    hf_cache = cache_dir or "/root/workspace/.hf_cache"
    lines.extend([
        f"mkdir -p {shlex.quote(hf_cache)}",
        f"export HF_HOME={shlex.quote(hf_cache)}",
        f"export TRANSFORMERS_CACHE={shlex.quote(hf_cache)}",
        f"export HF_HUB_CACHE={shlex.quote(hf_cache)}",
    ])

    # Install and run
    lines.extend([
        "echo '=== Installing dependencies ==='",
        "uv sync --python 3.12",
        "echo '=== Running command ==='",
        f"uv run --python 3.12 {command}",
        "echo '=== Done ==='",
    ])

    return "\n".join(lines)


def launch_runpod_job(
    *,
    command: str,
    runtime_yaml: str | Path,
    pod_name_prefix: str = "alpaca-eval",
    no_wait: bool = False,
    dry_run: bool = False,
) -> str | None:
    """Launch a command on a RunPod GPU pod.

    Args:
        command: The shell command to run inside the pod (after uv sync).
            Example: "python scripts/alpaca_eval/generate_outputs.py --model_names ll-3.1-8b"
        runtime_yaml: Path to RuntimeConfig YAML with RunPod settings.
        pod_name_prefix: Prefix for the timestamped pod name.
        no_wait: If True, return immediately after pod creation.
        dry_run: If True, print payload but don't create pod.

    Returns:
        Pod ID string, or None for dry runs.
    """
    runtime = load_runtime_config(str(runtime_yaml))

    repo_url = runtime.runpod.repo_url or infer_repo_url()
    repo_ref = runtime.runpod.repo_ref or infer_repo_ref()
    env = resolve_runpod_env(runtime)

    # Resolve network volume — set to "none" in config to explicitly disable.
    # Disabling greatly improves GPU availability (no data center restriction).
    nv_config = runtime.runpod.network_volume_id
    if nv_config == "none":
        network_volume_id = None
    elif nv_config:
        network_volume_id = nv_config
    else:
        network_volume_id = os.getenv("RUNPOD_NETWORK_VOLUME_ID")

    # Only use HF cache dir if network volume is attached (it persists across pods)
    cache_dir = runtime.local.cache_dir if network_volume_id else None

    startup_script = _build_startup_script(
        repo_url=repo_url,
        repo_ref=repo_ref,
        editable_deps=runtime.runpod.editable_deps,
        workspace_subdir=runtime.runpod.workspace_subdir,
        cache_dir=cache_dir,
        command=command,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    pod_name = f"{pod_name_prefix}-{timestamp}".replace("_", "-")

    payload = {
        "name": pod_name,
        "cloudType": runtime.runpod.cloud_type,
        "computeType": "GPU",
        "gpuCount": runtime.runpod.gpu_count,
        "gpuTypeIds": runtime.runpod.gpu_type_ids,
        "gpuTypePriority": "custom" if runtime.runpod.gpu_type_ids else "availability",
        "imageName": runtime.runpod.image_name,
        "containerDiskInGb": runtime.runpod.container_disk_gb,
        "ports": ["22/tcp"],
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [startup_script],
        "env": env,
    }
    # Only attach network volume if available — skipping it greatly improves
    # GPU availability since pods aren't restricted to the volume's data center.
    # Results are uploaded to HF instead.
    if network_volume_id:
        payload["volumeMountPath"] = runtime.runpod.volume_mount_path
        payload["networkVolumeId"] = network_volume_id

    if dry_run:
        # Redact secrets in env for display
        display_env = {
            k: "<redacted>" if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET")) else v
            for k, v in payload["env"].items()
        }
        display_payload = {**payload, "env": display_env}
        import json
        print(json.dumps(display_payload, indent=2))
        return None

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY environment variable required")

    client = RunPodClient(api_key)

    # Try configured cloud_type first, fall back to the other on failure
    primary = payload["cloudType"]
    fallback = "COMMUNITY" if primary == "SECURE" else "SECURE"
    cloud_types = [primary, fallback]

    pod = None
    for i, cloud_type in enumerate(cloud_types):
        payload["cloudType"] = cloud_type
        try:
            pod = client.create_pod(payload)
            break
        except RuntimeError as e:
            if "could not find any pods" in str(e):
                if i < len(cloud_types) - 1:
                    next_type = cloud_types[i + 1]
                    print(f"  No {cloud_type} GPUs available, trying {next_type}...")
                else:
                    print(f"  No {cloud_type} GPUs available either.")
                continue
            raise

    if pod is None:
        raise RuntimeError(
            f"No GPUs available on SECURE or COMMUNITY for types: {runtime.runpod.gpu_type_ids}"
        )

    pod_id = pod["id"]
    print(f"✓ Created pod {pod_id} ({pod_name}) on {payload['cloudType']}")

    if no_wait:
        print(f"  Pod running in background. Check status: runpodctl get pod {pod_id}")
        return pod_id

    print(f"  Waiting for pod to complete (polling every {runtime.runpod.poll_interval_seconds}s)...")
    final_pod = client.wait_for_exit(
        pod_id,
        poll_interval_seconds=runtime.runpod.poll_interval_seconds,
    )
    status = final_pod.get("desiredStatus", "unknown")
    print(f"  Pod {pod_id} finished with status: {status}")

    if runtime.runpod.terminate_on_exit:
        client.delete_pod(pod_id)
        print(f"  Deleted pod {pod_id}")

    return pod_id
