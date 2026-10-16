"""Round24 launcher with nullable-error feedback compatibility.

Keep the frozen numerical controller and journal implementation byte-identical:
existing run plans and checksummed assessments must remain valid on resume.
Only the controller's source-feedback projection is replaced; journal records,
grading, model code, fingerprints and cache validation are unchanged.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def source_feedback(path, limit=8):
    from vcell.evolution.journal import events
    from vcell.utils import file_sha256, write_json

    path = Path(path)
    records = [r for r in events(path) if r.get('phase') == 'search'][-limit:]
    result = []
    for record in records:
        error = record.get('error')
        if error is None:
            error = ''
        if not isinstance(error, str):
            raise TypeError('Research feedback error must be a string or null')
        result.append({
            'kind': record['kind'], 'category': record.get('category'),
            'error': error[-1600:], 'code_sha256': record.get('code_sha256'),
            'score': record.get('score'),
            'hypothesis_status': record.get('hypothesis_status'),
        })
    write_json(path.parent / 'feedback_compatibility.json', {
        'entrypoint_sha256': file_sha256(__file__),
        'journal_sha256': file_sha256(path) if path.exists() else None,
        'change': 'Normalize missing/null error to empty text in source feedback only',
        'journal_rewritten': False, 'numerical_fingerprint_overridden': False,
        'test_evaluated': False,
    })
    return result


def configure_controller(controller):
    # This imported binding is used only by source_memory, not the evaluator.
    controller.feedback = source_feedback
    return controller


def main():
    import run_round24

    controller = configure_controller(run_round24)
    print('ROUND24 FEEDBACK FIX: nullable errors normalized; frozen numerical '
          'code and assessment checksums retained', flush=True)
    controller.main()


if __name__ == '__main__':
    main()
