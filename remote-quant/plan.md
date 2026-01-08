# Plan: Remote Quantization Tool (remote-quant)

## Overview
Python CLI tool that orchestrates model quantization on remote NVIDIA GPUs from a local Intel Xeon host. Handles SSH connection, job submission, progress monitoring, weight compression, and transfer.

## Requirements
- Submit quantization jobs to remote GPU server via SSH
- Monitor job progress in real-time
- Compress quantized weights (tar + pigz)
- Transfer compressed weights back to local host
- Support self-SSH for local testing
- Track job state (pending, running, completed, failed)
- Configuration via YAML file

## Directory Structure
```
remote-quant/
├── pyproject.toml              # Package config with CLI entry point
├── config.example.yaml         # Example configuration
├── README.md                   # Usage documentation
└── remote_quant/
    ├── __init__.py
    ├── cli.py                  # Main CLI (click-based)
    ├── config.py               # Configuration loading
    ├── ssh.py                  # SSH connection management
    ├── job.py                  # Job submission and monitoring
    ├── transfer.py             # Compression and file transfer
    └── state.py                # Job state persistence
```

## Files to Create

### 1. `remote-quant/pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "remote-quant"
version = "0.1.0"
description = "Remote GPU quantization orchestrator"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
    "paramiko>=3.0",
    "pyyaml>=6.0",
    "rich>=13.0",
]

[project.scripts]
remote-quant = "remote_quant.cli:main"
```

### 2. `remote-quant/config.example.yaml`
```yaml
# Remote GPU host configuration
remote:
  host: "gpu-server.local"      # or localhost for self-SSH test
  user: "root"
  port: 22
  key_file: "~/.ssh/id_rsa"     # SSH private key path

# Remote paths
remote_paths:
  working_dir: "/root/auto-round"
  venv: "/root/auto-round/.venv"
  output_dir: "./quantized_models"

# Local paths
local:
  output_dir: "./models"
  state_file: "./remote-quant-state.json"

# Quantization defaults
quantization:
  scheme: "W4A16"
  format: "auto_round,auto_gptq"
  recipe: "light"               # light or default

# Transfer settings
transfer:
  compress: true
  compression_threads: 4        # pigz threads
  cleanup_remote: false         # delete remote files after transfer
```

### 3. `remote-quant/remote_quant/__init__.py`
```python
__version__ = "0.1.0"
```

### 4. `remote-quant/remote_quant/cli.py`
Main CLI with commands:
- `remote-quant submit <model>` - Submit quantization job
- `remote-quant status [job_id]` - Check job status
- `remote-quant logs <job_id>` - Stream remote logs
- `remote-quant fetch <job_id>` - Compress and transfer weights
- `remote-quant list` - List all jobs

### 5. `remote-quant/remote_quant/config.py`
- Load YAML configuration
- Merge with CLI overrides
- Validate required fields
- Expand paths (~, env vars)

### 6. `remote-quant/remote_quant/ssh.py`
- SSHConnection class using paramiko
- Execute commands with output streaming
- SCP file transfer
- Connection pooling/reuse

### 7. `remote-quant/remote_quant/job.py`
- Build quantization command
- Submit via nohup for background execution
- Poll for completion (check process, log file)
- Parse completion status from logs

### 8. `remote-quant/remote_quant/transfer.py`
- Remote compression (tar + pigz)
- SCP download with progress bar
- Local extraction
- Cleanup remote compressed files

### 9. `remote-quant/remote_quant/state.py`
- JSON-based job state persistence
- Track: job_id, model, status, timestamps, paths
- Query by status, job_id

## CLI Usage Examples

```bash
# Install
cd remote-quant && pip install -e .

# Submit a job
remote-quant submit meta-llama/Llama-3.1-8B --config config.yaml

# Check status
remote-quant status
remote-quant status job_20260108_123456

# Stream logs
remote-quant logs job_20260108_123456 --follow

# Fetch completed weights
remote-quant fetch job_20260108_123456

# Full pipeline (submit + wait + fetch)
remote-quant run meta-llama/Llama-3.1-8B --wait --fetch
```

## Workflow Diagram
```
[Local Xeon]                    [Remote GPU]
     |                               |
     |-- SSH: submit job ----------->|
     |                               |-- nohup auto-round-light
     |<-- job_id --------------------|
     |                               |
     |-- SSH: poll status ---------->|
     |<-- running/completed ---------|
     |                               |
     |-- SSH: tar + pigz ----------->|
     |                               |-- compress weights
     |<-- SCP: weights.tar.gz -------|
     |                               |
     |-- extract locally             |
```

## Testing Strategy
1. **Self-SSH test**: Configure `host: localhost` to test full flow
2. **Unit tests**: Mock SSH for job/transfer logic
3. **Integration**: Test against actual remote GPU

## Verification
```bash
# After implementation, test with:
cd /root/auto-round/remote-quant
pip install -e .

# Self-SSH test (localhost)
cp config.example.yaml config.yaml
# Edit config.yaml: set host to localhost

# Submit test job
remote-quant submit meta-llama/Llama-3.1-8B --config config.yaml

# Monitor
remote-quant status
remote-quant logs <job_id> --follow

# Fetch when complete
remote-quant fetch <job_id>
```
