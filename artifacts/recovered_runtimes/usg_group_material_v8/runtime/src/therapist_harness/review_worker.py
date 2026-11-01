"""Client for the isolated, persistent greedy review ASR process."""

import json
from pathlib import Path
import selectors
import subprocess
import sys
import time

from .store import sha256_file


class ReviewASRWorker:
    def __init__(self, *, base_model, adapter, session_path, device="cuda:0"):
        self.session_path = Path(session_path).resolve()
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.session_path.with_suffix(".log").open("x")
        self.started = time.perf_counter()
        self.process = subprocess.Popen([sys.executable,
            str(Path(__file__).resolve().parents[2] / "examples/transcribe_sg_review.py"),
            "--worker", "--output", str(self.session_path), "--base-model", str(base_model),
            "--adapter", str(adapter), "--device", device], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.closed = False
        try:
            ready = self._receive()
            if ready.get("status") != "ready" or ready.get("session_sha256") != sha256_file(self.session_path):
                raise RuntimeError("review ASR worker failed to initialize")
            self.startup_seconds = time.perf_counter() - self.started
        except BaseException:
            self.abort()
            raise

    def _receive(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=600):
                raise TimeoutError("review ASR worker response timeout")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("review ASR worker exited before its response; see session log")
        return json.loads(line)

    def transcribe(self, audio, output):
        output = Path(output).resolve()
        self.process.stdin.write(json.dumps({"audio": str(Path(audio).resolve()), "output": str(output)}) + "\n")
        self.process.stdin.flush()
        response = self._receive()
        if (response.get("status") != "complete" or response.get("output") != str(output)
                or response.get("sha256") != sha256_file(output)):
            raise RuntimeError("review ASR worker transcription failed")
        return output

    def close(self):
        if self.closed:
            return
        self.process.stdin.write('{"command":"close"}\n')
        self.process.stdin.flush()
        response = self._receive()
        code = self.process.wait(timeout=60)
        self.closed = True
        self.log.close()
        if code != 0 or response.get("status") != "complete":
            raise RuntimeError("review ASR worker final model verification failed")

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=60)
        self.closed = True
        self.log.close()
