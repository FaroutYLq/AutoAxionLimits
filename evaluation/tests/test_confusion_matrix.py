"""Phase 1c (#625) — per-coupling-type confusion matrix.

`build_coupling_type_confusion` is a pure function over the per-paper report
list, multi-type-aware (a prediction is correct iff it is in ANY of the paper's
GT types). It surfaces the confusable clusters the aggregate accuracy hides.

Run:  pytest evaluation/tests/test_confusion_matrix.py -v
"""

from __future__ import annotations

from evaluation.metrics import build_coupling_type_confusion


def _paper(pred, expected, correct=None):
    if correct is None:
        correct = pred in (expected if isinstance(expected, list) else [expected])
    return {
        "coupling_type_predicted": pred,
        "coupling_type_expected": expected,
        "coupling_type_correct": correct,
    }


def test_correct_single_type_diagonal():
    conf = build_coupling_type_confusion([_paper("AxionPhoton", ["AxionPhoton"])])
    assert conf["matrix"]["AxionPhoton"]["AxionPhoton"] == 1
    assert conf["confusions"] == []
    assert conf["n_graded"] == 1 and conf["n_correct"] == 1
    assert conf["accuracy"] == 1.0


def test_multi_type_predicted_in_set_is_correct():
    # GT has two types; predicting either is correct -> diagonal on the predicted.
    conf = build_coupling_type_confusion(
        [_paper("DarkPhoton", ["DarkPhoton", "VectorBL"])])
    assert conf["matrix"]["DarkPhoton"]["DarkPhoton"] == 1
    assert conf["confusions"] == []
    assert conf["n_correct"] == 1


def test_wrong_prediction_credits_every_gt_row():
    # Predicted DarkPhoton, GT is {VectorBL} -> off-diagonal VectorBL->DarkPhoton.
    conf = build_coupling_type_confusion(
        [_paper("DarkPhoton", ["VectorBL"], correct=False)])
    assert conf["matrix"]["VectorBL"]["DarkPhoton"] == 1
    assert conf["confusions"] == [
        {"gt": "VectorBL", "predicted": "DarkPhoton", "count": 1}]
    assert conf["n_correct"] == 0


def test_wrong_prediction_multi_gt_credits_all_rows():
    conf = build_coupling_type_confusion(
        [_paper("AxionPhoton", ["ScalarPhoton", "ScalarBaryon"], correct=False)])
    assert conf["matrix"]["ScalarPhoton"]["AxionPhoton"] == 1
    assert conf["matrix"]["ScalarBaryon"]["AxionPhoton"] == 1


def test_confusions_sorted_richest_first():
    papers = (
        [_paper("DarkPhoton", ["VectorBL"], correct=False)] * 3
        + [_paper("AxionProton", ["AxionNeutron"], correct=False)] * 1
    )
    conf = build_coupling_type_confusion(papers)
    assert conf["confusions"][0] == {
        "gt": "VectorBL", "predicted": "DarkPhoton", "count": 3}
    assert conf["confusions"][1]["count"] == 1


def test_skips_no_prediction_and_no_gt():
    papers = [
        _paper(None, ["AxionPhoton"]),          # no prediction
        _paper("DarkPhoton", []),                # no GT type
        _paper("AxionPhoton", ["AxionPhoton"]),  # graded
    ]
    conf = build_coupling_type_confusion(papers)
    assert conf["n_skipped"] == 2
    assert conf["n_graded"] == 1
    assert conf["accuracy"] == 1.0


def test_accuracy_reconciles_with_correct_flag():
    papers = [
        _paper("AxionPhoton", ["AxionPhoton"]),
        _paper("DarkPhoton", ["VectorBL"], correct=False),
        _paper("VectorBL", ["VectorBL", "DarkPhoton"]),
    ]
    conf = build_coupling_type_confusion(papers)
    assert conf["n_graded"] == 3
    assert conf["n_correct"] == 2
    assert conf["accuracy"] == 2 / 3
