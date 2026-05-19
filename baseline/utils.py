from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime

DEFAULT_LOG_ROOT = "output"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_ACTIVE_LOG_FILE = None
_ACTIVE_LOG_DIR = None
_ACTIVE_TASK_COUNTER = 0


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TeeStream:
    """Write evaluator terminal output to the console and a plain-text log."""

    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, data):
        self.console_stream.write(data)
        self.console_stream.flush()
        self.log_stream.write(strip_ansi(data))
        self.log_stream.flush()

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()


def strip_ansi(text: str) -> str:
    return ANSI_PATTERN.sub("", text)


def safe_filename(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe_value or "item"


def create_unique_dir(parent_dir: str, dirname: str) -> str:
    path = os.path.join(parent_dir, dirname)
    if not os.path.exists(path):
        os.makedirs(path)
        return path

    suffix = 1
    while True:
        candidate = os.path.join(parent_dir, f"{dirname}_{suffix:02d}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        suffix += 1


def relpath(path: str) -> str:
    return os.path.relpath(path, os.getcwd())


def write_text_file(path: str, content: str):
    with open(path, "w", encoding="utf8") as fp:
        fp.write(content)
        if content and not content.endswith("\n"):
            fp.write("\n")


def write_json_file(path: str, payload):
    with open(path, "w", encoding="utf8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


@contextmanager
def terminal_log(log_root: str = DEFAULT_LOG_ROOT):
    global _ACTIVE_LOG_FILE, _ACTIVE_LOG_DIR, _ACTIVE_TASK_COUNTER

    os.makedirs(log_root, exist_ok=True)
    started_at = datetime.now()
    run_id = f"run_{started_at.strftime('%Y%m%d_%H%M%S')}"
    log_dir = create_unique_dir(log_root, run_id)
    log_path = os.path.join(log_dir, "evaluator.log")
    run_metadata_path = os.path.join(log_dir, "run.json")
    run_metadata = {
        "run_id": os.path.basename(log_dir),
        "started_at": started_at.isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "argv": sys.argv,
        "terminal_log": "evaluator.log",
        "tasks_dir": "tasks",
        "status": "running",
    }
    write_json_file(run_metadata_path, run_metadata)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf8", buffering=1) as log_file:
        _ACTIVE_LOG_FILE = log_file
        _ACTIVE_LOG_DIR = log_dir
        _ACTIVE_TASK_COUNTER = 0
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(Colors.CYAN + f"Evaluator output directory: {log_dir}" + Colors.ENDC)
            print(Colors.CYAN + f"Evaluator terminal log: {log_path}" + Colors.ENDC)
            yield log_dir
            run_metadata["status"] = "completed"
        except Exception as exc:
            run_metadata["status"] = "failed"
            run_metadata["error"] = repr(exc)
            raise
        finally:
            finished_at = datetime.now()
            run_metadata["finished_at"] = finished_at.isoformat(timespec="seconds")
            run_metadata["duration_sec"] = round((finished_at - started_at).total_seconds(), 3)
            write_json_file(run_metadata_path, run_metadata)
            print(Colors.CYAN + f"Evaluator outputs saved to: {log_dir}" + Colors.ENDC)
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            _ACTIVE_LOG_FILE = None
            _ACTIVE_LOG_DIR = None


def log_only(message: str):
    if _ACTIVE_LOG_FILE is not None:
        _ACTIVE_LOG_FILE.write(strip_ansi(message))
        _ACTIVE_LOG_FILE.flush()


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def next_task_log_dir(task: str) -> str:
    global _ACTIVE_TASK_COUNTER

    if _ACTIVE_LOG_DIR is None:
        return None

    _ACTIVE_TASK_COUNTER += 1
    task_dir = os.path.join(_ACTIVE_LOG_DIR, "tasks", f"{_ACTIVE_TASK_COUNTER:02d}_{safe_filename(task)}")
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


def log_subprocess_result(task: str, result: subprocess.CompletedProcess, command, started_at=None, finished_at=None, task_dir: str = None):
    if task_dir is None:
        task_dir = next_task_log_dir(task)
    if task_dir is None:
        return None

    stdout = to_text(result.stdout)
    stderr = to_text(result.stderr)
    command_text = shlex.join(command)
    stdout_path = os.path.join(task_dir, "stdout.log")
    stderr_path = os.path.join(task_dir, "stderr.log")
    command_path = os.path.join(task_dir, "command.txt")
    result_path = os.path.join(task_dir, "result.json")

    write_text_file(stdout_path, stdout)
    write_text_file(stderr_path, stderr)
    write_text_file(command_path, command_text)
    write_json_file(result_path, {
        "task": task,
        "command": command,
        "returncode": result.returncode,
        "started_at": started_at.isoformat(timespec="seconds") if started_at else None,
        "finished_at": finished_at.isoformat(timespec="seconds") if finished_at else None,
        "duration_sec": round((finished_at - started_at).total_seconds(), 3) if started_at and finished_at else None,
        "stdout": "stdout.log",
        "stderr": "stderr.log",
    })

    log_only("\n" + "=" * 80 + "\n")
    log_only(f"[{datetime.now().isoformat(timespec='seconds')}] TASK LOG: {task}\n")
    log_only(f"Directory: {relpath(task_dir)}\n")
    log_only(f"Command: {command_text}\n")
    log_only(f"Return code: {result.returncode}\n")
    log_only(f"stdout: {relpath(stdout_path)}\n")
    log_only(f"stderr: {relpath(stderr_path)}\n")
    log_only(f"metadata: {relpath(result_path)}\n")
    log_only("=" * 80 + "\n")

    return task_dir


def write_task_summary(task_dir: str, summary):
    if task_dir is not None:
        write_json_file(os.path.join(task_dir, "summary.json"), summary)


def write_run_summary(results, total_elapsed: float):
    if _ACTIVE_LOG_DIR is not None:
        write_json_file(os.path.join(_ACTIVE_LOG_DIR, "summary.json"), {
            "total_elapsed_sec": total_elapsed,
            "tasks": results,
        })
