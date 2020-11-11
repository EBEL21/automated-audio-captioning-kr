#!/usr/bin/env python

from pathlib import Path
import json
import pandas as pd
from eval_metrics import evaluate_metrics_from_files, write_json

__author__ = 'Samuel Lipping -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['test']


def create_score_json(pred_file, ref_file, score_file):
    metrics, per_file_metrics = evaluate_metrics_from_files(pred_file, ref_file)
    scores = []
    for k, metric_dict in per_file_metrics.items():
        SPICE_score = metric_dict['SPICE']['All']['f']
        metric_dict.pop('SPICE')
        metric_dict['SPICE'] = SPICE_score
        scores.append(metric_dict)
    write_json(scores, score_file)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    pred_file = Path('tmp/601967_2020-11-05_20-22-29_756491_pred.json')
    ref_file = Path('tmp/601967_2020-11-05_20-22-29_756491_ref.json')
    score_file = Path('tmp/601967_2020-11-05_20-22-29_756491_score.json')
    result_file = Path('tmp/601967_2020-11-05_20-22-29_756491_result.csv')

    create_score_json(pred_file, ref_file, score_file)

    pred = pd.read_json(pred_file, encoding="UTF-8")
    pred.rename(columns={"caption": "caption_pred"}, inplace=True)
    with ref_file.open("r") as ref_f:
        ref = ref_f.read()
    ref = json.loads(ref)['annotations']
    ref = pd.DataFrame(ref).rename(columns={"caption": "caption_ref"})
    score = pd.read_json(score_file)
    m1 = pd.merge(pred, ref, on='audio_id')
    m1 = pd.merge(m1, score, on='audio_id')
    m1.drop(columns=["id"], inplace=True)
    # df = m1.set_index(['audio_id', 'caption_pred', 'caption_ref'])
    m1.to_csv(result_file)
    print(m1[:10])
