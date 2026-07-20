from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_LOG_REQUESTS = os.environ.get("NEMO_SKILLS_SANDBOX_LOG_REQUESTS", "0") == "1"


def _set_default_env() -> None:
    # Avoid CPU oversubscription when running user code/evaluators inside the sandbox.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")

    # Prefer node-local scratch for temp files when available.
    if "TMPDIR" not in os.environ and os.environ.get("SLURM_TMPDIR"):
        os.environ["TMPDIR"] = os.environ["SLURM_TMPDIR"]


def _set_limits() -> None:
    # Match NeMo's convention: interpret env var as bytes and apply a 2x headroom.
    mem_limit_bytes = int(os.environ.get("NEMO_SKILLS_SANDBOX_MEM_LIMIT", 50 * 1024**3))
    hard = 2 * mem_limit_bytes

    # Best-effort RLIMITs; some clusters disallow lowering certain limits.
    for rlimit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
        try:
            resource.setrlimit(rlimit, (hard, hard))
        except (OSError, ValueError):
            pass

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            return


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _execute_shell(
    command: str,
    *,
    timeout_s: float,
    std_input: str | None,
    max_output_characters: int,
) -> dict[str, Any]:
    started = time.time()
    tmp_root = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or None
    with tempfile.TemporaryDirectory(prefix="lcb_sandbox_", dir=tmp_root) as td:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=Path(td),
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )  # noqa: S603, S607
        try:
            stdout, stderr = proc.communicate(input=std_input or "", timeout=timeout_s)
            status = "completed"
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            stdout, stderr = "", "TIMEOUT"
            status = "timeout"
        except Exception as e:
            _kill_process_group(proc)
            stdout, stderr = "", f"ERROR: {e}"
            status = "error"

    elapsed = time.time() - started
    return {
        "process_status": status,
        "return_code": proc.returncode,
        "stdout": _truncate(stdout or "", max_output_characters),
        "stderr": _truncate(stderr or "", max_output_characters),
        "time": elapsed,
    }


class SandboxHandler(BaseHTTPRequestHandler):
    server_version = "livecodebench-local-sandbox/0.1"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
        # Keep logs minimal by default (Slurm job logs can get huge).
        if _LOG_REQUESTS:
            super().log_message(fmt, *args)
        return

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        if path == "/health":
            self._send_json(200, {"status": "healthy"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        if path != "/execute":
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            req = json.loads(body)
        except Exception:
            self._send_json(400, {"error": "invalid json"})
            return

        language = (req.get("language") or "shell").lower()
        if language != "shell":
            self._send_json(400, {"error": f"unsupported language: {language}"})
            return

        cmd = str(req.get("generated_code") or "")
        timeout_s = float(req.get("timeout") or 30)
        max_out = int(
            req.get("max_output_characters")
            or os.environ.get("NEMO_SKILLS_SANDBOX_MAX_OUTPUT_CHARACTERS", "100_000")
        )
        std_input = req.get("std_input")
        if std_input is not None:
            std_input = str(std_input)

        result = _execute_shell(
            cmd,
            timeout_s=timeout_s,
            std_input=std_input,
            max_output_characters=max_out,
        )
        self._send_json(200, result)


def main() -> None:
    bind = os.environ.get("NEMO_SKILLS_SANDBOX_BIND", "127.0.0.1")
    port = int(os.environ.get("NEMO_SKILLS_SANDBOX_PORT", "6000"))

    _set_default_env()
    _set_limits()
    print(f"[LCB sandbox server] listening on http://{bind}:{port} (pid={os.getpid()})", flush=True)
    httpd = ThreadingHTTPServer((bind, port), SandboxHandler)
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
