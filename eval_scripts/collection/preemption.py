"""Stop only this evaluation's subprocesses when an interrupt is received."""
import os
import signal
import subprocess


class Preempted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'Evaluation interrupted by signal {signum}')


class PreemptionSupervisor:
    """Forward a stop to the current process group, then let the evaluation exit."""

    def __init__(self):
        self.signum = None
        self.process = None
        self.handlers = {}

    def __enter__(self):
        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.handlers[signum] = signal.signal(signum, self._request_stop)
        return self

    def __exit__(self, *exc):
        for signum, handler in self.handlers.items():
            signal.signal(signum, handler)

    def _request_stop(self, signum, frame):
        self.signum = self.signum or signum
        if self.process is not None:
            self._signal_group(signal.SIGTERM)

    def _signal_group(self, signum):
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            pass

    def check(self):
        if self.signum is not None:
            raise Preempted(self.signum)

    def run(self, command, **kwargs):
        self.check()
        self.process = subprocess.Popen(command, start_new_session=True, **kwargs)
        try:
            while True:
                if self.signum is not None:
                    self._signal_group(signal.SIGTERM)
                    try:
                        self.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self._signal_group(signal.SIGKILL)
                        self.process.wait()
                    self.check()
                try:
                    code = self.process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            self.check()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            self.process = None
