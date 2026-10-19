"""Build the licensed 2026-10-19 research release from a verified server export.

No exported Python is executed. Checkpoints use PyTorch's restricted weights-only
loader. Public artifacts exclude the server inventory and original absolute paths.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import torch


ROOT = Path(__file__).resolve().parents[1]
QWEN_SHA = '37ad93d5f3d736a816317b983fc02fa3387933eae2288867591e293afc6e2cf4'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')


def build(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    folder = destination / 'virtual-cell-research-models'
    folder.mkdir()
    models = []
    with tarfile.open(archive, 'r:gz') as stream:
        names = stream.getnames()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for member in stream.getmembers():
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                raise ValueError('Unsafe archive member')
        index = json.load(stream.extractfile('ARCHIVE_MANIFEST.json'))['files']
        if set(names) != set(index) | {'ARCHIVE_MANIFEST.json'}:
            raise ValueError('Archive manifest membership mismatch')
        for name, expected in index.items():
            data = stream.extractfile(name).read()
            if len(data) != expected['bytes'] or sha(data) != expected['sha256']:
                raise ValueError('Server export checksum mismatch: ' + name)
        entries = json.load(stream.extractfile('selected_models.json'))
        snapshot = json.load(stream.extractfile('snapshot.json'))
        for entry in entries:
            data = stream.extractfile(entry['package_path']).read()
            if sha(data) != entry['sha256']:
                raise ValueError('Selected model checksum mismatch')
            if entry['original_path'] == 'qwen_421_four_teacher_01/student_a/all/best.pt':
                if sha(data) != QWEN_SHA:
                    raise ValueError('This release is pinned to the reviewed historical Student A')
                saved = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
                if saved['format'] != 'vcell-qwen-adapter-v1' or len(saved['genes']) != 2000 or len(saved['perturbations']) != 196:
                    raise ValueError('Unexpected published adapter format or vocabulary')
                if not all(torch.isfinite(v).all() for v in saved['trainable_state'].values()):
                    raise ValueError('Nonfinite adapter parameter')
                model = {'id': 'qwen_student_a', 'kind': 'qwen_adapter', 'file': 'qwen_student_a/adapter.pt',
                         'sha256': entry['sha256'], 'bytes': len(data), 'genes': len(saved['genes']),
                         'perturbations': len(saved['perturbations']), 'best_epoch': saved['epoch'],
                         'trainable_parameters': sum(v.numel() for v in saved['trainable_state'].values()),
                         'trainable_tensors': len(saved['trainable_state']), 'base_included': False,
                         'state_distillation_terms': 'licenses/MODEL_LICENSE.md', 'independent_validation': False}
                config = {**model, 'checkpoint_format': saved['format'],
                          'model_spec': {**saved['model_spec'], 'model_id': 'Qwen/Qwen3-0.6B-Base',
                                         'revision': 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'},
                          'backbone_file_sha256': saved['backbone_identity'],
                          'normalization': {'mean': 'Saved per-gene training control mean',
                                            'std': 'Saved per-gene training control standard deviation',
                                            'scale': float(saved['normalization']['scale'])},
                          'training_data_fingerprint': saved['data_fingerprint'],
                          'training_run_fingerprint': saved['run_fingerprint'],
                          'raw_preprocessing_audit_included': False}
                dump(folder / 'qwen_student_a/config.json', config)
                (folder / 'qwen_student_a/genes.txt').write_text('\n'.join(saved['genes']) + '\n', encoding='utf-8')
                (folder / 'qwen_student_a/perturbations.txt').write_text('\n'.join(saved['perturbations']) + '\n', encoding='utf-8')
            elif entry['run'] == 'round26_critic_02' and entry['package_path'].endswith('/module.py'):
                parts = Path(entry['original_path']).parts
                arm, task = parts[2:4]
                tree = ast.parse(data.decode('utf-8'))
                if sum(isinstance(n, ast.FunctionDef) and n.name == 'fit_predict' for n in tree.body) != 1:
                    raise ValueError('Invalid algorithm entry point')
                if not entry.get('completion_checksum_matched') or not entry.get('assessment_checksum_matched'):
                    raise ValueError('Unverified frozen algorithm')
                model = {'id': arm + '_' + task + '_' + entry['sha256'][:10], 'kind': 'evolved_algorithm',
                         'task': task, 'arm': arm, 'file': f"methods/{entry['sha256']}.py",
                         'sha256': entry['sha256'], 'bytes': len(data), 'genes': 1809,
                         'selection': 'existing source-only frozen shortlist; includes diagnostic Pareto candidates',
                         'source_templates_included': False, 'independent_validation': False}
            else:
                raise ValueError('Unrecognized model in this versioned release')
            path = folder / model['file']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            models.append(model)
    if len(models) != 8 or len({m['id'] for m in models}) != 8:
        raise ValueError('Expected one adapter and seven distinct frozen methods')
    dump(folder / 'manifest.json', {'format': 'vcell-research-release-v1', 'release': 'research-models-2026-10-19',
         'models': models, 'server_export_sha256': sha(Path(archive).read_bytes()),
         'export_source_commit': snapshot['source_code']['commit'], 'round27_status': 'unfinished; not included',
         'standalone_foundation_model': False, 'independent_validation': False})
    (folder / 'licenses').mkdir()
    for name in ('MODEL_LICENSE.md', 'MODEL_ACCEPTABLE_USE_POLICY.md', 'STATE_CODE_LICENSE.txt', 'APACHE_2_0.txt'):
        shutil.copyfile(ROOT / 'licenses' / name, folder / 'licenses' / name)
    shutil.copyfile(ROOT / 'LICENSE', folder / 'licenses/REPOSITORY_CODE_MIT.txt')
    shutil.copyfile(ROOT / 'THIRD_PARTY_NOTICES.md', folder / 'THIRD_PARTY_NOTICES.md')
    shutil.copyfile(ROOT / 'docs/MODEL_RELEASE_20261019.md', folder / 'README.md')
    (folder / 'MODEL_CARD.md').write_text(
        '# Virtual Cell research artifacts\n\n'
        'This is an archival research release, not a validated deployment model.\n'
        'The Qwen Student A checkpoint is State-distilled and remains subject to the complete\n'
        'Arc Research Institute State Model Non-Commercial License in licenses/MODEL_LICENSE.md.\n'
        'Repository MIT licensing does not remove these restrictions. The frozen Qwen base is not included.\n\n'
        'State citation: Adduri, A. et al. (2025). Predicting cellular responses to perturbation\n'
        'across diverse contexts with State. https://doi.org/10.1101/2025.06.26.661135\n\n'
        'Seven source-frozen Round26 methods require legal public source templates and refitting.\n'
        'They are not seven trained neural checkpoints and are not all validated winners.\n'
        'Round27 is unfinished and excluded. Read README.md and manifest.json for configuration,\n'
        'inference contracts, limitations and exact file checksums. No raw expression data are included.\n', encoding='utf-8')
    dump(folder / 'validation_report.json', {
        'server_export_payloads_verified': len(index), 'export_model_files_verified': len(models),
        'restricted_checkpoint_load': 'torch.load(weights_only=True)',
        'checkpoint_tensors_finite': True, 'algorithm_ast_entrypoints_valid': 7,
        'full_original_runs_verified': False, 'biological_predictions_reproduced': False,
        'checkpoint_historical_sha256': QWEN_SHA,
        'missing_dependencies': ['Qwen base snapshot', 'query/source inputs', 'original full preprocessing audit'],
        'source_code': 'See the matching GitHub release tag for inference code and software tests'})
    files = {p.relative_to(folder).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
             for p in sorted(folder.rglob('*')) if p.is_file()}
    dump(folder / 'ARCHIVE_MANIFEST.json', {'files': files})
    result = destination / 'virtual-cell-research-models-20261019.tar.gz'
    with tarfile.open(result, 'w:gz') as stream:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                stream.add(path, arcname=Path(folder.name) / path.relative_to(folder), recursive=False)
    checksum = sha(result.read_bytes())
    (destination / 'SHA256SUMS').write_text(checksum + '  ' + result.name + '\n', encoding='utf-8')
    print(json.dumps({'archive': str(result), 'bytes': result.stat().st_size, 'sha256': checksum}, indent=2))
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.export_archive, args.output)
