"""Atomic progress records shared by collection and detector evaluation."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


class AtomicJSON:
    @staticmethod
    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as out:
                temporary = Path(out.name)
                json.dump(value, out)
                out.flush()
                os.fsync(out.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def read(path):
        try:
            with Path(path).open() as stream:
                value = json.load(stream)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def file_digest(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def evaluator_digest():
        paths = [ROOT / 'evaluate_all_methods.py', *sorted((ROOT / 'detectors').rglob('*.py'))]
        return AtomicJSON.digest({str(p.relative_to(ROOT)): AtomicJSON.file_digest(p) for p in paths})


class EvaluationProgress:
    """Save raw scores per sample/layer; recompute dataset-wide aggregates at the end."""

    def __init__(self, args):
        settings = vars(args).copy()
        settings.pop('resume', None)
        for name in ('rep_stiff_combined_weights', 'rep_stiff_lara_clean_ref'):
            if settings.get(name):
                settings[name] = AtomicJSON.file_digest(settings[name])
        settings['evaluator_digest'] = AtomicJSON.evaluator_digest()
        self.directory = Path(args.output_summary_json).parent / '.evaluation_progress' / AtomicJSON.digest(settings)

    @staticmethod
    def key(data_item):
        return AtomicJSON.digest(data_item)

    def load(self, key):
        return AtomicJSON.read(self.directory / f'{key}.json')

    def save(self, key, scores, representation=None, complete=False):
        AtomicJSON.write(self.directory / f'{key}.json', {
            'scores': scores, 'representation': representation, 'complete': complete,
        })


class CompletionRecord:
    """Skip finished checkpoints only while their settings and artifacts still match."""

    def __init__(self, path, config, scoring, artifacts):
        self.path = Path(path)
        self.artifacts = [Path(p) for p in artifacts]
        self.identity = AtomicJSON.digest({
            'config': config, 'scoring': scoring, 'evaluator_digest': AtomicJSON.evaluator_digest(),
            'generator_digest': AtomicJSON.file_digest(ROOT / 'generate_full_data.py'),
        })

    def _hashes(self):
        return {str(path): AtomicJSON.file_digest(path) for path in self.artifacts}

    def complete(self):
        record = AtomicJSON.read(self.path)
        if not record or record.get('identity') != self.identity:
            return False
        try:
            return record.get('artifacts') == self._hashes()
        except OSError:
            return False

    def save(self):
        AtomicJSON.write(self.path, {'identity': self.identity, 'artifacts': self._hashes()})
