"""Recompute scores from new run predictions and compare ordered evaluation IDs."""
import csv
import json
from pathlib import Path
from .scoring import score_response_r1

def run(config, paths, args):
    reference_ids = None
    rows = []
    for run_dir in args.runs:
        folder = paths.within(run_dir)
        with (folder / 'predictions.jsonl').open(encoding='utf-8') as stream:
            predictions = [json.loads(line) for line in stream if line.strip()]
        ids = [p['id'] for p in predictions]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f'Empty or duplicate prediction IDs: {run_dir}')
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f'Evaluation ID order differs: {run_dir}')
        recalculated = [score_response_r1(p['response'], str(p['gold']), p['truncated']) for p in predictions]
        for old, fresh in zip(predictions, recalculated):
            if any((old[key] != fresh[key] for key in ('correct', 'correct_strict'))):
                raise ValueError(f"Stored score mismatch: {run_dir}, {old['id']}")
        n = len(predictions)
        correct = sum((r['correct'] for r in recalculated))
        strict = sum((r['correct_strict'] for r in recalculated))
        rows.append(
            {
                'model': Path(run_dir).name,
                'n_questions': n,
                'n_correct': correct,
                'accuracy_r1': correct / n,
                'n_correct_strict': strict,
                'accuracy_strict': strict / n,
            },
        )
    output = paths.within(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0
