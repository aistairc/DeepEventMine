# -*- coding: utf-8 -*-
# © Khoa Duong (dnanhkhoa@live.com)
from collections import defaultdict

import texttable


def precision(tp=0, fp=0):
    if tp + fp:
        return tp / (tp + fp)
    return 0.0


def recall(tp=0, fn=0):
    if tp + fn:
        return tp / (tp + fn)
    return 0.0


def f_score(precision, recall, beta=1.0):
    """
    The beta parameter determines the weight of precision in the combined score. beta < 1 lends more weight to precision, while beta > 1 favors recall (beta -> 0 considers only precision, beta -> inf only recall).
    """
    numerator = (1 + beta ** 2) * precision * recall
    denominator = beta ** 2 * precision + recall
    if denominator:
        return numerator / denominator
    return 0.0


def specificity(tn=0, fp=0):
    # FPR
    if tn + fp:
        return tn / (tn + fp)
    return 0.0


def sensitivity(tp=0, fn=0):
    # TPR
    return recall(tp, fn)


def auc(specificity, sensitivity):
    return (specificity + sensitivity) / 2


def measure(tp=0, tn=0, fp=0, fn=0, beta=1.0, lenient=False):
    if lenient and tp + fn == 0:
        tp, tn, fp, fn = 1, 1, 0, 0

    _precision = precision(tp, fp)
    _recall = recall(tp, fn)
    _f_score = f_score(_precision, _recall, beta)
    _specificity = specificity(tn, fp)
    _sensitivity = sensitivity(tp, fn)
    _auc = auc(_specificity, _sensitivity)

    return {
        "precision": _precision * 100,
        "recall": _recall * 100,
        "f_score": _f_score * 100,
        "specificity": _specificity * 100,
        "sensitivity": _sensitivity * 100,
        "auc": _auc * 100,
    }


def count(pred_entities, gold_entities, label):
    assert label, "Label is invalid"

    # Remove duplicates
    pred_entities = {e: True for e in pred_entities}
    gold_entities = {e: True for e in gold_entities}

    positions = {**pred_entities, **gold_entities}

    padded_pred_entities, padded_gold_entities = [], []

    for k in positions:
        if k in pred_entities and k[-1] == label:
            padded_pred_entities.append(k[-1])
        else:
            padded_pred_entities.append(None)

        if k in gold_entities and k[-1] == label:
            padded_gold_entities.append(k[-1])
        else:
            padded_gold_entities.append(None)

    matches = list(zip(padded_pred_entities, padded_gold_entities))

    return {
        "tp": matches.count((label, label)),
        "tn": matches.count((None, None)),
        "fp": matches.count((label, None)),
        "fn": matches.count((None, label)),
    }


def eval_nner(preds, golds, labels, beta=1.0, lenient=False):
    num_pred_sentences = len(preds)
    num_gold_sentences = len(golds)
    assert num_pred_sentences == num_gold_sentences

    all_scores = []
    counts = [defaultdict(int) for _ in labels]

    for sentence_id in range(num_gold_sentences):
        pred_entities = set(preds[sentence_id])
        gold_entities = set(golds[sentence_id])

        for label_id, label in enumerate(labels):
            for k, v in count(pred_entities, gold_entities, label).items():
                counts[label_id][k] += v

    tt = texttable.Texttable()
    tt.set_cols_width([28] + [10] * 6 + [10] * 3)
    tt.set_cols_dtype(["t", "f", "f", "f", "f", "f", "f", "i", "i", "i"])
    tt.set_cols_align(["l"] * 10)
    tt.header(
        [
            "Labels",
            "Prec.",
            "Rec.",
            "F(b={})".format(beta),
            "Speci.",
            "Sensi.",
            "AUC",
            "Pred.",
            "Gold.",
            "Corr.",
        ]
    )

    total_counts = defaultdict(int)

    for label_id, label in enumerate(labels):
        score = measure(
            counts[label_id]["tp"],
            counts[label_id]["tn"],
            counts[label_id]["fp"],
            counts[label_id]["fn"],
            beta,
            lenient,
        )

        total_counts["tp"] += counts[label_id]["tp"]
        total_counts["tn"] += counts[label_id]["tn"]
        total_counts["fp"] += counts[label_id]["fp"]
        total_counts["fn"] += counts[label_id]["fn"]

        all_scores.append(
            [
                label,
                score["precision"],
                score["recall"],
                score["f_score"],
                score["specificity"],
                score["sensitivity"],
                score["auc"],
                counts[label_id]["tp"] + counts[label_id]["fp"],
                counts[label_id]["tp"] + counts[label_id]["fn"],
                counts[label_id]["tp"],
            ]
        )
        tt.add_row(all_scores[-1])

    score = measure(
        total_counts["tp"],
        total_counts["tn"],
        total_counts["fp"],
        total_counts["fn"],
        beta,
        lenient,
    )

    all_scores.append(
        [
            "Overall",
            score["precision"],
            score["recall"],
            score["f_score"],
            score["specificity"],
            score["sensitivity"],
            score["auc"],
            total_counts["tp"] + total_counts["fp"],
            total_counts["tp"] + total_counts["fn"],
            total_counts["tp"],
        ]
    )
    tt.add_row(all_scores[-1])

    return tt.draw(), all_scores


if __name__ == "__main__":
    labels = ["A", "B", "C", "D", "E"]  # DO NOT INCLUDE "O"
    golds = [
        [(0, 2, "E"), (0, 2, "A"), (1, 3, "B"), (4, 6, "D")],
        [(0, 1, "C"), (3, 4, "A")],
        [(2, 3, "B"), (3, 4, "A"), (4, 5, "C"), (4, 5, "A")],
    ]
    preds = [
        [(0, 2, "E"), (0, 2, "A"), (1, 3, "C"), (4, 6, "D"), (4, 6, "D")],
        [(0, 1, "C"), (3, 4, "A")],
        [(2, 3, "B"), (3, 4, "A"), (4, 5, "D"), (4, 5, "A")],
    ]
    res, score = eval_nner(preds, golds, labels)
    print(res)
    print(score)
