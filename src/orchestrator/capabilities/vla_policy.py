import os
import time
import json
import yaml
import logging
import threading
import subprocess
from pathlib import Path
from typing import Any
from .camera import ZMQ_HOST, ZMQ_PORT

logger = logging.getLogger(__name__)


def _load_policies() -> dict[str, dict[str, str]]:
    policy_path = Path(__file__).parent / "policies.yaml"
    if not policy_path.exists():
        logger.warning("policies.yaml not found at %s. VLA policies unavailable.", policy_path)
        return {}
    with open(policy_path, "r") as f:
        return yaml.safe_load(f) or {}

POLICIES: dict[str, dict[str, str]] = _load_policies()


_vla_process: subprocess.Popen | None = None
_vla_ready_time: float | None = None
_VLA_SERVER         = os.getenv("VLA_INFERENCE_SERVER", "192.168.50.42:8080")
_ROBOT_PORT         = os.getenv("ROBOT_PORT", "/dev/ttyACM0")
_ROBOT_ID           = os.getenv("ROBOT_ID", "left_arm")
_VLA_START_TIMEOUT  = int(os.getenv("VLA_START_TIMEOUT", "300"))  # seconds
# Fires only after client.start() completes (model loaded on server) and both
# the action-receiver thread AND the control-loop thread have crossed their
# threading.Barrier — meaning the robot is about to send its first observation.
_VLA_READY_SIGNAL = "Action receiving thread starting"
_vla_log_path: Path | None = None   # path to the current policy's log file


def _watch_log_file(log_path: Path, process: subprocess.Popen, ready_event: threading.Event, signal_seen: list[bool], output_buf: list[str]) -> None:
    """Tail the log file until the ready signal appears or the process exits.

    Writing subprocess output to a file (instead of a PIPE) is critical: a PIPE
    has a 64 KB kernel buffer and blocks the subprocess write() syscall when full.
    If the orchestrator's log handler is slow (e.g. lock contention with the agent),
    the buffer fills up and the robot client freezes — no more observations sent,
    no more action chunks generated.  A file never blocks the writer.
    """
    with open(log_path, "r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                if process.poll() is not None:
                    ready_event.set()
                    return
                time.sleep(0.05)
                continue
            line = line.rstrip()
            if line:
                logger.info("[robot_client] %s", line)
                output_buf.append(line)
            if _VLA_READY_SIGNAL in line:
                signal_seen[0] = True
                ready_event.set()
                return # stop watching; subprocess continues writing to the file on its own


def start_vla_policy_fn(policy_id: str, **kwargs: Any) -> dict[str, Any]:
    global _vla_process, _vla_log_path, _vla_ready_time

    if _vla_process is not None and _vla_process.poll() is None:
        return {"status": "FAILURE", "error": "A VLA policy is already running. Call stop_vla_policy first.", "pid": _vla_process.pid}
    
    _vla_ready_time = None

    policy_cfg = POLICIES.get(policy_id)
    if not policy_cfg:
        return {"status": "FAILURE", "error": f"Policy '{policy_id}' not found. Available: {list(POLICIES.keys())}"}

    camera_config = json.dumps({
        "base": {"type": "zmq", "server_address": ZMQ_HOST, "port": ZMQ_PORT, "camera_name": "base", "width": 640, "height": 480, "fps": 30},
        "wrist": {"type": "zmq", "server_address": ZMQ_HOST, "port": ZMQ_PORT, "camera_name": "wrist", "width": 640, "height": 480, "fps": 30},
    })

    cmd = [
        "python", "-u", "-m", "lerobot.async_inference.robot_client",
        "--robot.type=so101_follower", f"--robot.port={_ROBOT_PORT}",
        f"--robot.id={_ROBOT_ID}", f"--robot.cameras={camera_config}",
        f"--task={policy_cfg['task']}", f"--server_address={_VLA_SERVER}",
        f"--policy_type={policy_cfg['policy_type']}",
        f"--pretrained_name_or_path={policy_cfg['repo_id']}",
        "--policy_device=cuda", "--client_device=cuda", "--actions_per_chunk=25",
    ]

    # Write output to a file, NOT a PIPE.  A PIPE has a finite kernel buffer
    # (~64 KB on Linux); if the orchestrator is slow reading it, the subprocess
    # blocks inside write() and the entire robot control loop freezes.
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    _vla_log_path = log_dir / f"vla_client_{int(time.time())}.log"
    log_file = open(_vla_log_path, "wb")

    try:
        _vla_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log_file, stderr=log_file)
        log_file.close() # orchestrator only needs the read side
        # Robot's _follower.py calls input() on startup to confirm calibration.
        # Send ENTER to auto-accept; without this the process hangs forever.
        _vla_process.stdin.write(b"\n")
        _vla_process.stdin.flush()
    except Exception as exc:
        log_file.close() 
        return {"status": "FAILURE", "error": str(exc)}

    ready_event = threading.Event()
    signal_seen = [False]
    output_buf: list[str] = []

    threading.Thread(target=_watch_log_file, args=(_vla_log_path, _vla_process, ready_event, signal_seen, output_buf), daemon=True).start()

    ready_event.wait(timeout=_VLA_START_TIMEOUT)

    if signal_seen[0]:
        _vla_ready_time = time.monotonic()

    def _tail(n: int = 30) -> str:
        lines = output_buf[-n:] if output_buf else []
        return "\n".join(lines) if lines else "(no output captured)"

    if not signal_seen[0]:
        rc = _vla_process.poll()
        if rc is not None:
            err = f"Policy '{policy_id}' process exited with code {rc} before inference started."
        else:
            _vla_process.terminate()
            try:
                _vla_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _vla_process.kill()
                _vla_process.wait()
            err = f"Policy '{policy_id}' did not start within {_VLA_START_TIMEOUT}s."
        _vla_process = None
        return {"status": "FAILURE", "error": err, "output_tail": _tail(), "log_file": str(_vla_log_path)}

    if _vla_process.poll() is not None:
        _vla_process = None
        return {"status": "FAILURE", "error": f"Policy exited immediately.", "output_tail": _tail(), "log_file": str(_vla_log_path)}

    return {
        "status": "SUCCESS",
        "message": f"Policy '{policy_id}' is running (task: \"{policy_cfg['task']}\").",
        "pid": _vla_process.pid,
        "log_file": str(_vla_log_path),
        "typical_wait_time": policy_cfg.get("typical_wait_time", 30),
    }


def stop_vla_policy_fn(**kwargs: Any) -> dict[str, Any]:
    global _vla_process
    if _vla_process is None or _vla_process.poll() is not None:
        _vla_process = None
        return {"status": "WARNING", "message": "No VLA policy was running."}

    _vla_process.terminate()
    try:
        _vla_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _vla_process.kill()
        _vla_process.wait()

    elapsed: float | None = None
    if _vla_ready_time is not None:
        elapsed = round(time.monotonic() - _vla_ready_time, 2)

    _vla_process = None
    result: dict[str, Any] = {"status": "SUCCESS", "message": "VLA policy stopped."}
    if elapsed is not None:
        result["policy_run_duration_seconds"] = elapsed
    return result


def wait_fn(seconds: int, **kwargs: Any) -> dict[str, Any]:
    time.sleep(int(seconds))
    return {"status": "SUCCESS", "message": f"Waited {seconds} seconds."}