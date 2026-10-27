#!/usr/bin/env python3
"""Evaluate a pinned LaRA collection checkpoint, or the whole collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('models.json')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval_scripts.collection.checkpoints import CompletionRecord
from eval_scripts.collection.preemption import Preempted, PreemptionSupervisor
DETECTORS = ('ppl', 'mink', 'mink_plus_plus', 'recall', 'cdd',
             'self_critique', 'self_critique_ablation', 'lara', 'self_critique_lara')
GENERATION_METHODS = {
    'recall': 'dime', 'cdd': 'consistency',
    'self_critique_lara': 'self_critique',
    'self_critique': 'self_critique', 'self_critique_ablation': 'self_critique_ablation',
}
RESULT_FIELDS = {'dime': 'perturbed_greedy_results', 'consistency': 'original_random_results',
                 'self_critique': 'critique_greedy_results',
                 'self_critique_ablation': 'unfamiliar_greedy_results'}



def requires_lara(detectors):
    return bool({'lara', 'self_critique_lara'} & set(detectors))


class CollectionCLI:
    """Build the existing command-line interface without changing its defaults."""

    @staticmethod
    def build_parser():
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument('--model', default='all', help='Collection model name or repo ID; default: all')
        p.add_argument('--family', choices=['eurus', 'limr'])
        p.add_argument('--task', choices=['kk', 'sat'])
        p.add_argument('--epoch', type=int, choices=[0, 1, 2])
        p.add_argument('--detectors', nargs='+', choices=['all', *DETECTORS], default=['all'])
        p.add_argument('--list', action='store_true', help='List matching checkpoints and exit')
        p.add_argument('--dry-run', action='store_true', help='Print commands without downloading or running anything')
        p.add_argument('--prepare-only', action='store_true', help='Prepare labeled inputs without loading a model')
        p.add_argument('--stage', choices=['full', 'generate', 'evaluate'], default='full')
        p.add_argument('--data-file', type=Path, help='Override the matching labeled KK/SAT input (.parquet/.jsonl)')
        p.add_argument('--samples-per-class', type=int, default=0, help='0: all rows; otherwise sample this many members and nonmembers')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--resume', action='store_true', help='Reuse completed model and per-sample evaluation checkpoints')
        p.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16', help='Generation precision; use float16 for GPUs without bfloat16 support')
        p.add_argument('--output-root', type=Path, default=ROOT / 'final_results/collection')
        p.add_argument('--max-tokens', type=int, default=2048)
        p.add_argument('--batch-size', type=int, default=8)
        p.add_argument('--tensor-parallel-size', type=int, default=1)
        p.add_argument('--gpu-memory-utilization', type=float, default=.85)
        p.add_argument('--max-model-len', type=int)
        p.add_argument('--num-random-samples', type=int, default=10)
        p.add_argument('--temperature-random', type=float, default=.8)
        p.add_argument('--approx-mode', choices=['renorm', 'rest'], default='renorm')
        p.add_argument('--openrouter-workers', type=int, default=4)
        p.add_argument('--num-blanks', type=int, default=1)
        p.add_argument('--clean-reference', type=Path, help='Optional held-out LaRA reference statistics')
        return p


class ModelCatalog:
    """Select checkpoints from a pinned collection manifest."""

    def __init__(self, manifest):
        self.manifest = manifest

    def select(self, args):
        models = [m for m in self.manifest['models'] if
                  (args.model == 'all' or args.model in (m['repo_id'], m['repo_id'].split('/')[-1]))
                  and (args.family is None or m['family'] == args.family)
                  and (args.task is None or m['task'] == args.task)
                  and (args.epoch is None or m['epoch'] == args.epoch)]
        if not models:
            raise ValueError('No collection checkpoint matches the requested filters; use --list')
        return models


class JsonLines:
    """Write evaluation records using the existing replacement convention."""

    @staticmethod
    def write(path, rows):
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        temporary.replace(path)


class BenchmarkDataset:
    """Load, validate and sample one explicitly labeled benchmark."""

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args

    @staticmethod
    def normalize_rows(frame, samples_per_class=0, seed=42):
        """Preserve explicit labels; never infer membership from a split name."""
        import pandas as pd
        if not {'prompt', 'member'}.issubset(frame.columns):
            raise ValueError('Evaluation data requires prompt and explicit member columns')
        if frame.member.isna().any() or not frame.member.isin([0, 1, False, True]).all():
            raise ValueError('member must contain explicit 0/1 labels')
        if set(frame.member.astype(int)) != {0, 1}:
            raise ValueError('Evaluation requires both members and nonmembers')
        rows, seen = [], {}
        for row in frame.to_dict('records'):
            messages = row['prompt']
            if hasattr(messages, 'tolist'):
                messages = messages.tolist()
            if not isinstance(messages, list) or not messages or any(
                not isinstance(m, dict) or not isinstance(m.get('content'), str)
                or m.get('role') not in {'system', 'user', 'assistant'} for m in messages
            ):
                raise ValueError('prompt must be a nonempty list of role/content messages')
            users = [m['content'] for m in messages if m['role'] == 'user']
            if messages[-1]['role'] != 'user':
                raise ValueError('Evaluation prompts must end with a user message')
            if not users or not users[-1].strip():
                raise ValueError('Every prompt must have nonempty user content')
            key = users[-1]
            label = int(row['member'])
            if key in seen:
                if seen[key] != label:
                    raise ValueError('The same prompt has conflicting membership labels')
                continue  # The generation cache identifies prompts by their user content.
            seen[key] = label
            source = row.get('data_source', 'unknown')
            rows.append({'prompt': messages, 'member': label,
                         'data_source': 'unknown' if pd.isna(source) else str(source)})
        if samples_per_class:
            result = pd.DataFrame(rows)
            if min(result.member.value_counts()) < samples_per_class:
                raise ValueError('samples-per-class exceeds the available unique members/nonmembers')
            rows = pd.concat([group.sample(samples_per_class, random_state=seed)
                              for _, group in result.groupby('member')]).to_dict('records')
        return rows

    def prepare(self, target):
        dataset, args = self.dataset, self.args
        import pandas as pd
        if args.data_file:
            path = args.data_file.resolve()
            frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_json(path, lines=True)
        else:
            path = ROOT / dataset['benchmark_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != dataset['benchmark_sha256']:
                raise ValueError(f'Bundled benchmark changed: {path}; use --data-file to explicitly override it')
            frame = pd.read_parquet(path)
        rows = self.normalize_rows(frame, args.samples_per_class, args.seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        JsonLines.write(target, rows)
        counts = {label: sum(row['member'] == label for row in rows) for label in (0, 1)}
        print(f'Input: {target} ({counts[1]} members, {counts[0]} nonmembers)', flush=True)
        return rows


class RunConfiguration:
    """Represent the settings that identify reusable generation artifacts."""

    def __init__(self, model, dataset, args):
        self.model = model
        self.dataset = dataset
        self.args = args

    def as_dict(self):
        model, dataset, args = self.model, self.dataset, self.args
        data = {'dataset': dataset}
        if args.data_file:
            data = {'path': str(args.data_file.resolve()),
                    'sha256': hashlib.sha256(args.data_file.read_bytes()).hexdigest()}
        return {'model': model, 'input': data, 'samples_per_class': args.samples_per_class,
                'seed': args.seed, 'max_tokens': args.max_tokens, 'num_random_samples': args.num_random_samples,
                'temperature_random': args.temperature_random, 'approx_mode': args.approx_mode,
                'max_model_len': args.max_model_len, 'dtype': args.dtype, 'generation_schema': 2}


class EvaluationCommandBuilder:
    """Build generation and scoring commands for the selected detectors."""

    def __init__(self, args, detectors):
        self.args = args
        self.detectors = detectors

    def build(self, model_path, data_file, generated, score_dir, layers, cache_dir):
        args, detectors = self.args, self.detectors
        methods = sorted({GENERATION_METHODS[d] for d in detectors if d in GENERATION_METHODS})
        generation = [sys.executable, str(ROOT / 'generate_full_data.py'),
            '--model_path', str(model_path), '--data_root_dir', str(data_file),
            '--output_file', str(generated), '--prompt_template', 'own',
            '--max_tokens', str(args.max_tokens), '--batch_size', str(args.batch_size),
            '--tensor_parallel_size', str(args.tensor_parallel_size),
            '--gpu_memory_utilization', str(args.gpu_memory_utilization),
            '--num_random_samples', str(args.num_random_samples),
            '--temperature_random', str(args.temperature_random), '--approx_mode', args.approx_mode,
            '--dtype', args.dtype,
            '--methods_to_run', *methods]
        if args.max_model_len:
            generation += ['--max_model_len', str(args.max_model_len)]
        evaluation = [sys.executable, str(ROOT / 'evaluate_all_methods.py'),
            '--input_file', str(generated), '--output_summary_json', str(score_dir / 'evaluation_summary.json'),
            '--output_plot', str(score_dir / 'performance_plot.png'), '--detectors', *detectors]
        if requires_lara(detectors):
            evaluation += ['--rep_stiff_model_name', str(model_path), '--rep_stiff_layers', layers,
                '--rep_stiff_max_workers', str(args.openrouter_workers),
                '--rep_stiff_output_dir', str(cache_dir), '--rep_stiff_combined_fixed',
                '--rep_stiff_incomplete_num_blanks', str(args.num_blanks)]
            if args.clean_reference:
                evaluation += ['--rep_stiff_lara_clean_ref', str(args.clean_reference.resolve())]
        if args.resume:
            evaluation.append('--resume')
        return generation, evaluation


class GeneratedDataValidator:
    """Check that a cached generation contains each requested detector input."""

    def __init__(self, detectors):
        self.detectors = detectors

    def validate(self, path, expected_rows=None):
        detectors = self.detectors
        required = {'original_greedy_results'} if set(detectors) - {'lara'} else set()
        required.update(RESULT_FIELDS[GENERATION_METHODS[d]] for d in detectors if d in GENERATION_METHODS)
        count = 0
        observed = []
        with path.open() as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line)
                observed.append((row.get('original_user_content'), row.get('ground_truth_label'), row.get('data_source')))
                missing = [key for key in required if not row.get(key)]
                if missing:
                    raise ValueError(f'{path}, row {count}: missing {missing}; rerun --stage generate')
        if not count:
            raise ValueError(f'No generated rows in {path}')
        if expected_rows is not None:
            expected = [(next(m['content'] for m in reversed(row['prompt']) if m['role'] == 'user'),
                         row['member'], row['data_source']) for row in expected_rows]
            if len(observed) != len(expected) or set(observed) != set(expected):
                raise ValueError(f'{path}: generated rows do not match all benchmark inputs; rerun --stage generate')


class ModelEvaluationRun:
    """Coordinate the input, generation and scoring stages for one checkpoint."""

    def __init__(self, model, dataset, args, detectors, supervisor=None):
        self.supervisor = supervisor
        self.model = model
        self.dataset = dataset
        self.args = args
        self.detectors = detectors
        self.benchmark = BenchmarkDataset(dataset, args)
        self.configuration = RunConfiguration(model, dataset, args)
        self.command_builder = EvaluationCommandBuilder(args, detectors)
        self.validator = GeneratedDataValidator(detectors)

    def _run_command(self, command, env):
        if self.supervisor is None:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        else:
            self.supervisor.run(command, cwd=ROOT, env=env)

    def run(self):
        model, dataset, args, detectors = self.model, self.dataset, self.args, self.detectors
        config = self.configuration.as_dict()
        run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        directory = args.output_root.resolve() / model['repo_id'].split('/')[-1] / run_id
        data_file = directory / 'evaluation_input.jsonl'
        generated = directory / ('representation_input.jsonl' if detectors == ['lara'] else 'generated_data.jsonl')
        scoring = {'detectors': detectors, 'num_blanks': args.num_blanks,
                   'clean_reference': (hashlib.sha256(args.clean_reference.read_bytes()).hexdigest()
                                       if args.clean_reference else None)}
        score_id = hashlib.sha256(json.dumps(scoring, sort_keys=True).encode()).hexdigest()[:8]
        score_dir = directory / ('+'.join(detectors) + '-' + score_id)
        cache_dir = args.output_root.resolve() / 'perturbation_cache' / model['task']
        model_path = f'<snapshot:{model["repo_id"]}@{model["revision"]}>'
        print(f'\nModel: {model["repo_id"]}\nResults: {score_dir}', flush=True)
        if args.dry_run:
            print(f'Prepare {dataset["benchmark_path"] if not args.data_file else args.data_file} -> {data_file}')
            generation, evaluation = self.command_builder.build(model_path, data_file, generated, score_dir,
                '<all transformer layers from config.json>', cache_dir)
            if args.stage != 'evaluate':
                print('Write representation input (no generation)' if detectors == ['lara'] else shlex.join(generation))
            if args.stage != 'generate':
                print(shlex.join(evaluation))
            return
        rows = self.benchmark.prepare(data_file)
        (directory / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
        if args.prepare_only:
            return
        artifacts = [data_file, generated, score_dir / 'evaluation_summary.json',
                     score_dir / 'all_method_scores.jsonl', score_dir / 'method_directions.json']
        if requires_lara(detectors):
            artifacts.append(score_dir / 'rep_stiff_scores.json')
        completion = CompletionRecord(score_dir / '.completed.json', config, scoring, artifacts)
        if args.resume and args.stage != 'generate' and completion.complete():
            print('Already completed; skipping this checkpoint.', flush=True)
            return
        if detectors == ['lara']:
            JsonLines.write(generated, ({'original_user_content': next(m['content'] for m in reversed(row['prompt']) if m['role'] == 'user'),
                                     'ground_truth_label': row['member'], 'data_source': row['data_source']} for row in rows))
        need_model = (args.stage != 'evaluate' and detectors != ['lara']) or (requires_lara(detectors) and args.stage != 'generate')
        layers = ''
        if need_model:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model['repo_id'], revision=model['revision'])
            model_config = json.loads((Path(model_path) / 'config.json').read_text())
            layers = ','.join(f'L{i}' for i in range(model_config['num_hidden_layers']))
        generation, evaluation = self.command_builder.build(model_path, data_file, generated, score_dir, layers, cache_dir)
        env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn')
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
        if args.stage != 'evaluate' and detectors != ['lara']:
            self._run_command(generation, env)
        if args.stage != 'generate':
            self.validator.validate(generated, rows)
            score_dir.mkdir(parents=True, exist_ok=True)
            (score_dir / 'detectors.json').write_text(json.dumps(scoring, indent=2) + '\n')
            self._run_command(evaluation, env)
            if self.supervisor:
                self.supervisor.check()
            completion.save()


class CollectionApplication:
    """Apply collection-wide options and execute the selected checkpoint runs."""

    def __init__(self):
        self.parser = CollectionCLI.build_parser()

    def run(self):
        p = self.parser
        args = p.parse_args()
        try:
            manifest = json.loads(MANIFEST.read_text())
            models = ModelCatalog(manifest).select(args)
            if args.list:
                for model in models:
                    print(model['repo_id'])
                return
            for name in ('max_tokens', 'batch_size', 'tensor_parallel_size', 'num_random_samples', 'openrouter_workers', 'num_blanks'):
                if getattr(args, name) < 1:
                    raise ValueError(f'--{name.replace("_", "-")} must be positive')
            if args.samples_per_class < 0:
                raise ValueError('--samples-per-class must be nonnegative')
            if not 0 < args.gpu_memory_utilization <= 1:
                raise ValueError('--gpu-memory-utilization must be in (0, 1]')
            detectors = [d for d in DETECTORS if 'all' in args.detectors or d in args.detectors]
            if requires_lara(detectors) and args.stage != 'generate' and not (args.dry_run or args.prepare_only):
                try:
                    from dotenv import load_dotenv
                    load_dotenv(ROOT / '.env')
                except ImportError:
                    pass
                if not os.environ.get('OPENROUTER_API_KEY'):
                    raise ValueError('LaRA needs OPENROUTER_API_KEY; choose baseline detectors to run without it')
            with PreemptionSupervisor() as supervisor:
                for model in models:
                    supervisor.check()
                    ModelEvaluationRun(model, manifest['datasets'][model['task']], args, detectors, supervisor).run()
        except Preempted as exc:
            import signal
            print('Evaluation interrupted; completed samples are saved for --resume.', flush=True)
            return 75 if exc.signum == signal.SIGUSR1 else 128 + exc.signum
        except (ValueError, FileNotFoundError) as exc:
            p.error(str(exc))

# Preserve the importable function API used by existing callers.
def parser():
    return CollectionCLI.build_parser()


def selected_models(manifest, args):
    return ModelCatalog(manifest).select(args)


def normalize_rows(frame, samples_per_class=0, seed=42):
    return BenchmarkDataset.normalize_rows(frame, samples_per_class, seed)


def prepare_input(dataset, args, target):
    return BenchmarkDataset(dataset, args).prepare(target)


def write_jsonl(path, rows):
    return JsonLines.write(path, rows)


def run_config(model, dataset, args):
    return RunConfiguration(model, dataset, args).as_dict()


def commands(args, model_path, data_file, generated, score_dir, layers, detectors, cache_dir):
    return EvaluationCommandBuilder(args, detectors).build(
        model_path, data_file, generated, score_dir, layers, cache_dir)


def validate_generated(path, detectors):
    return GeneratedDataValidator(detectors).validate(path)


def run_model(model, dataset, args, detectors):
    return ModelEvaluationRun(model, dataset, args, detectors).run()


def main():
    return CollectionApplication().run()


if __name__ == '__main__':
    raise SystemExit(main())
