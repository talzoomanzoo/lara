"""Coordinate perturbation-cache writers across independent Slurm jobs."""
import fcntl
from pathlib import Path


class SharedCacheLock:
    """Advisory lock released automatically if a worker is preempted or killed."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a')
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, *exc):
        self.stream.close()
