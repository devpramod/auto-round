#!/usr/bin/env python3
"""
Remote Quantization Demo

A minimal demo showing the vision:
1. Read config from YAML
2. Connect to remote GPU via SSH
3. Submit quantization job (background with nohup)
4. Stream logs with --watch
5. Compress and transfer weights

Usage:
    python demo.py config.yaml meta-llama/Llama-3.1-8B --dry-run
    python demo.py config.yaml meta-llama/Llama-3.1-8B --watch
    python demo.py config.yaml meta-llama/Llama-3.1-8B --skip-transfer
"""

import argparse
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_ssh_args(cfg: dict, background: bool = False) -> list:
    """Build SSH argument list from config."""
    conn = cfg["connection"]
    ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no"]

    # -f: fork to background after auth (for fire-and-forget commands)
    if background:
        ssh_args.append("-f")

    if conn.get("key_file"):
        key_path = Path(conn["key_file"]).expanduser()
        ssh_args.extend(["-i", str(key_path)])

    if conn.get("port", 22) != 22:
        ssh_args.extend(["-p", str(conn["port"])])

    # Proxy/jump host support
    if conn.get("proxy"):
        ssh_args.extend(["-o", f"ProxyCommand={conn['proxy']}"])

    target = f"{conn['user']}@{conn['host']}"
    ssh_args.append(target)

    return ssh_args


def ssh_cmd(cfg: dict, command: str, capture: bool = False, stream: bool = False) -> subprocess.CompletedProcess:
    """Execute command on remote host via SSH."""
    ssh_args = get_ssh_args(cfg)
    ssh_args.append(command)

    if capture:
        return subprocess.run(ssh_args, capture_output=True, text=True)
    elif stream:
        # Stream output in real-time
        process = subprocess.Popen(
            ssh_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
        except KeyboardInterrupt:
            process.terminate()
            print("\n[Interrupted]")
        process.wait()
        return subprocess.CompletedProcess(ssh_args, process.returncode)
    else:
        return subprocess.run(ssh_args)


def scp_download(cfg: dict, remote_path: str, local_path: str) -> bool:
    """Download file from remote host via SCP."""
    conn = cfg["connection"]
    scp_args = ["scp", "-o", "StrictHostKeyChecking=no"]

    if conn.get("key_file"):
        key_path = Path(conn["key_file"]).expanduser()
        scp_args.extend(["-i", str(key_path)])

    if conn.get("port", 22) != 22:
        scp_args.extend(["-P", str(conn["port"])])

    if conn.get("proxy"):
        scp_args.extend(["-o", f"ProxyCommand={conn['proxy']}"])

    source = f"{conn['user']}@{conn['host']}:{remote_path}"
    scp_args.extend(["-r", source, local_path])

    result = subprocess.run(scp_args)
    return result.returncode == 0


def expand_remote_path(path: str) -> str:
    """Expand ~ in path for remote shell (returns shell expression)."""
    if path.startswith("~/"):
        return f"$HOME/{path[2:]}"
    return path


def check_remote_setup(cfg: dict) -> bool:
    """Check if remote environment is set up with auto-round installed."""
    paths = cfg["remote_paths"]
    venv = expand_remote_path(paths["venv"])

    # Check if venv exists and auto-round is installed
    check_cmd = f"""
VENV_PATH="{venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    python -c "import auto_round" 2>/dev/null && echo "OK" || echo "MISSING"
else
    echo "NO_VENV"
fi
"""
    result = ssh_cmd(cfg, check_cmd, capture=True)
    status = result.stdout.strip().split('\n')[-1]
    return status == "OK"


def setup_remote_environment(cfg: dict) -> bool:
    """Set up remote environment with auto-round."""
    paths = cfg["remote_paths"]
    working_dir = expand_remote_path(paths["working_dir"])
    venv = expand_remote_path(paths["venv"])

    print("=" * 60)
    print("SETTING UP REMOTE ENVIRONMENT")
    print("=" * 60)
    print(f"Working dir: {paths['working_dir']}")
    print(f"Venv path:   {paths['venv']}")
    print("=" * 60)

    setup_cmd = f"""
set -e
WORKING_DIR="{working_dir}"
VENV_PATH="{venv}"

echo "Creating directory: $WORKING_DIR"
mkdir -p "$WORKING_DIR"
cd "$WORKING_DIR"

echo "Creating virtual environment..."
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing auto-round..."
pip install auto-round

echo "Installing lm-eval..."
pip install lm-eval

echo "Verifying installation..."
python -c "import auto_round; print(f'auto-round version: {{auto_round.__version__}}')"

echo ""
echo "Setup complete!"
"""

    print("\nRunning setup (this may take a few minutes)...\n")
    result = ssh_cmd(cfg, setup_cmd, stream=True)

    return result.returncode == 0


def ensure_remote_setup(cfg: dict) -> bool:
    """Ensure remote environment is ready, setup if needed."""
    print("Checking remote environment...")

    if check_remote_setup(cfg):
        print("Remote environment OK")
        return True

    print("Remote environment not ready, setting up...")
    return setup_remote_environment(cfg)


def build_quant_command(cfg: dict, model: str, job_id: str, background: bool = False) -> tuple[str, str]:
    """Build the quantization command. Returns (command, log_file_path)."""
    paths = cfg["remote_paths"]
    quant = cfg["quantization"]
    model_cfg = cfg.get("model", {})

    working_dir = expand_remote_path(paths["working_dir"])
    venv = expand_remote_path(paths["venv"])

    # Determine recipe command
    recipe = quant.get("recipe", "light")
    cmd = "auto-round-light" if recipe == "light" else "auto-round"

    # Build output directory path
    output_dir = f"{paths['output_dir']}/{job_id}"
    log_file = f"{paths.get('log_dir', './logs')}/quant_{job_id}.log"

    # Build environment exports
    env_exports = []
    if model_cfg.get("hf_token"):
        env_exports.append(f"export HF_TOKEN='{model_cfg['hf_token']}'")

    env_str = " && ".join(env_exports) + " && " if env_exports else ""

    # Build quantization command with optional iters
    iters_arg = f"--iters {quant['iters']}" if quant.get("iters") else ""

    quant_cmd = f"""{cmd} \\
    --model {model} \\
    --scheme {quant['scheme']} \\
    --format "{quant['format']}" \\
    {iters_arg} \\
    --output_dir {output_dir}"""

    if background:
        # Use nohup + disown for background execution (disown prevents SSH from waiting)
        full_cmd = f"""
cd {working_dir} && \\
source {venv}/bin/activate && \\
mkdir -p {paths.get('log_dir', './logs')} {paths['output_dir']} && \\
{env_str}nohup bash -c '{quant_cmd} 2>&1 | tee {log_file}' > /dev/null 2>&1 & disown
echo "Job started in background"
echo "Log file: {log_file}"
"""
    else:
        full_cmd = f"""
cd {working_dir} && \\
source {venv}/bin/activate && \\
mkdir -p {paths.get('log_dir', './logs')} {paths['output_dir']} && \\
{env_str}{quant_cmd} 2>&1 | tee {log_file}
"""

    return full_cmd.strip(), log_file


def submit_job(cfg: dict, model: str, job_id: str, dry_run: bool = False, watch: bool = False) -> bool:
    """Submit quantization job to remote host."""
    # If watching, run in background then tail logs
    background = watch
    cmd, log_file = build_quant_command(cfg, model, job_id, background=background)

    conn = cfg["connection"]
    print("=" * 60)
    print("REMOTE QUANTIZATION JOB")
    print("=" * 60)
    print(f"Job ID:  {job_id}")
    print(f"Model:   {model}")
    print(f"Host:    {conn['user']}@{conn['host']}:{conn.get('port', 22)}")
    print(f"Recipe:  {cfg['quantization'].get('recipe', 'light')}")
    print(f"Scheme:  {cfg['quantization']['scheme']}")
    print(f"Log:     {log_file}")
    print("=" * 60)

    if dry_run:
        print("\nCommand to execute:")
        print("-" * 60)
        print(cmd)
        print("-" * 60)
        print("\n[DRY RUN] Would execute above command on remote host")
        return True

    print("\nSubmitting job...")

    if watch:
        # Submit in background (ssh -f returns immediately), then stream logs
        ssh_args = get_ssh_args(cfg, background=True) + [cmd]
        subprocess.run(ssh_args)
        print("Job submitted")

        # Give it a moment to start
        time.sleep(2)

        # Stream logs
        print("\n" + "=" * 60)
        print("STREAMING LOGS (Ctrl+C to detach)")
        print("=" * 60 + "\n")

        paths = cfg["remote_paths"]
        working_dir = expand_remote_path(paths["working_dir"])
        output_dir = f"{paths['output_dir']}/{job_id}"
        if output_dir.startswith("./"):
            output_dir = output_dir[2:]

        # Use tail --pid to auto-exit when job completes, then show completion message
        tail_cmd = f"""cd {working_dir} && \\
PID=$(pgrep -f "auto-round.*{job_id}" | head -1) ; \\
tail -f --pid=$PID {log_file} 2>/dev/null ; \\
echo "" ; \\
echo "Saving model to disk..." ; \\
sleep 3 ; \\
echo "" ; \\
echo "============================================================" ; \\
echo "COMPLETE! Model saved to:" ; \\
echo "  {working_dir}/{output_dir}/" ; \\
echo "============================================================"
"""
        ssh_cmd(cfg, tail_cmd, stream=True)

        # tail --pid exits when job finishes, so we know it's done
        return True
    else:
        # Run in foreground
        result = ssh_cmd(cfg, cmd)
        return result.returncode == 0


def check_job_status(cfg: dict, job_id: str) -> str:
    """Check if job is still running on remote."""
    cmd = f"pgrep -f 'auto-round.*{job_id}' > /dev/null && echo 'running' || echo 'stopped'"
    result = ssh_cmd(cfg, cmd, capture=True)
    return result.stdout.strip()


def stream_logs(cfg: dict, job_id: str):
    """Stream logs for a running job."""
    paths = cfg["remote_paths"]
    working_dir = expand_remote_path(paths["working_dir"])
    log_file = f"{paths.get('log_dir', './logs')}/quant_{job_id}.log"
    output_dir = f"{paths['output_dir']}/{job_id}"
    if output_dir.startswith("./"):
        output_dir = output_dir[2:]

    print(f"Streaming logs from: {log_file}")
    print("=" * 60)
    print("(Ctrl+C to stop)")
    print("=" * 60 + "\n")

    # Use tail --pid to auto-exit when job completes
    tail_cmd = f"""cd {working_dir} && \\
PID=$(pgrep -f "auto-round.*{job_id}" | head -1) ; \\
tail -f --pid=$PID {log_file} 2>/dev/null ; \\
echo "" ; \\
echo "Saving model to disk..." ; \\
sleep 3 ; \\
echo "" ; \\
echo "============================================================" ; \\
echo "COMPLETE! Model saved to:" ; \\
echo "  {working_dir}/{output_dir}/" ; \\
echo "============================================================"
"""
    ssh_cmd(cfg, tail_cmd, stream=True)


def compress_weights(cfg: dict, job_id: str) -> str:
    """Compress quantized weights on remote host."""
    paths = cfg["remote_paths"]
    working_dir = expand_remote_path(paths["working_dir"])
    transfer = cfg.get("transfer", {})
    threads = transfer.get("compression_threads", 4)

    output_dir = f"{paths['output_dir']}/{job_id}"
    archive = f"{output_dir}.tar.gz"

    # Use pigz if available, fallback to gzip
    cmd = f"""
cd {working_dir} && \\
if command -v pigz &> /dev/null; then
    tar -cf - {output_dir} | pigz -p {threads} > {archive}
else
    tar -czf {archive} {output_dir}
fi && \\
echo "Compressed to: {archive}" && \\
ls -lh {archive}
"""

    print(f"\nCompressing weights on remote...")
    result = ssh_cmd(cfg, cmd)

    if result.returncode == 0:
        return archive
    return ""


def transfer_weights(cfg: dict, job_id: str, remote_archive: str) -> bool:
    """Transfer compressed weights to local host."""
    paths = cfg["remote_paths"]
    # For SCP, we need to resolve the full path on remote
    # Use a command to echo the actual path
    working_dir_raw = paths["working_dir"]
    if working_dir_raw.startswith("~/"):
        # For SCP, ~ is expanded by the remote shell
        full_remote_path = f"{working_dir_raw}/{remote_archive}"
    else:
        full_remote_path = f"{working_dir_raw}/{remote_archive}"

    local_dir = Path(cfg["local"]["output_dir"]).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / f"{job_id}.tar.gz"

    print(f"\nTransferring weights to {local_path}...")
    success = scp_download(cfg, full_remote_path, str(local_path))

    if success:
        print(f"Transfer complete: {local_path}")
        # Extract
        print("Extracting...")
        subprocess.run(["tar", "-xzf", str(local_path), "-C", str(local_dir)])
        print(f"Extracted to: {local_dir}/{job_id}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Remote Quantization Demo")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("model", nargs="?", help="Model to quantize (overrides config model_id)")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--job-id", help="Custom job ID (default: auto-generated)")
    parser.add_argument("--watch", action="store_true", help="Submit job and stream logs in real-time")
    parser.add_argument("--logs", metavar="JOB_ID", help="Stream logs for existing job")
    parser.add_argument("--skip-transfer", action="store_true", help="Skip weight transfer")
    parser.add_argument("--iters", type=int, help="Override iterations (lower = faster)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Stream logs for existing job
    if args.logs:
        stream_logs(cfg, args.logs)
        return 0

    # Get model from args or config
    model = args.model or cfg.get("model", {}).get("model_id")
    if not model:
        parser.error("model is required (via argument or config model.model_id)")

    # Override iters if specified
    if args.iters:
        cfg["quantization"]["iters"] = args.iters

    # Generate job ID
    job_id = args.job_id or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Ensure remote environment is set up (skip for dry-run)
    if not args.dry_run:
        if not ensure_remote_setup(cfg):
            print("Failed to setup remote environment!")
            return 1
        print()

    # Submit job
    result = submit_job(cfg, model, job_id, dry_run=args.dry_run, watch=args.watch)

    if args.dry_run:
        return 0

    if result == "running":
        # User detached but job still running - don't show completion
        return 0

    if not result:
        print("\nJob failed!")
        return 1

    # Build output path info
    paths = cfg["remote_paths"]
    working_dir = paths['working_dir']
    output_dir = paths['output_dir']
    # Clean up relative paths (./output -> output)
    if output_dir.startswith("./"):
        output_dir = output_dir[2:]
    remote_output = f"{working_dir}/{output_dir}/{job_id}"

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nQuantized weights stored on remote server:")
    print(f"  {remote_output}/")
    print()

    # Transfer weights
    if not args.skip_transfer and cfg.get("transfer", {}).get("compress", True):
        archive = compress_weights(cfg, job_id)
        if archive:
            local_dir = Path(cfg["local"]["output_dir"]).expanduser()
            transfer_weights(cfg, job_id, archive)
            print("\n" + "=" * 60)
            print("TRANSFER COMPLETE!")
            print("=" * 60)
            print(f"\nQuantized weights available locally at:")
            print(f"  {local_dir.absolute()}/output/{job_id}/")
            print()
    else:
        print("Skipping transfer. Use --skip-transfer=false to download weights.")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
