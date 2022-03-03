import pytest


from collections import Counter
from meshed import FuncNode
from meshed.dag import DAG
from pytest import fixture


def X_test(train_test_split):
    return train_test_split[1]


def y_test(train_test_split):
    return train_test_split[3]


def truth(y_test):  # to link up truth and test_y
    return y_test


def confusion_count(prediction, truth):
    """Get a dict containing the counts of all combinations of predicction and corresponding truth values."""
    return Counter(zip(prediction, truth))


def prediction(predict_proba, threshold):
    """Get an array of predictions from thresholding the scores of predict_proba array."""
    return list(map(lambda x: x >= threshold, predict_proba))


def predict_proba(model, X_test):
    """Get the prediction_proba scores of a model given some test data"""
    return model.predict_proba(X_test)


def _aligned_items(a, b):
    """Yield (k, a_value, b_value) triples for all k that are both a key of a and of b"""
    # reason for casting to dict is to make sure things like pd.Series use the right keys.
    # could also use k in a.keys() etc. to solve this.
    a = dict(a)
    b = dict(b)
    for k in a:
        if k in b:
            yield k, a[k], b[k]


def dot_product(a, b):
    """
    >>> dot_product({'a': 1, 'b': 2, 'c': 3}, {'b': 4, 'c': -1, 'd': 'whatever'})
    5
    """
    return sum(ak * bk for _, ak, bk in _aligned_items(a, b))


def classifier_score(confusion_count, confusion_value):
    """Compute a score for a classifier that produced the `confusion_count`, based on the given `confusion_value`.
    Meant to be curried by fixing the confusion_value dict.

    The function is purposely general -- it is not specific to binary classifier outcomes, or even any classifier outcomes.
    It simply computes a normalized dot product, depending on the inputs keys to align values to multiply and
    considering a missing key as an expression of a null value.
    """
    return dot_product(confusion_count, confusion_value) / sum(confusion_count.values())


@fixture
def bigger_dag():
    bigger_dag = DAG(
        [
            classifier_score,
            confusion_count,
            prediction,
            predict_proba,
            X_test,
            y_test,
            truth,
        ]
    )
    return bigger_dag


def test_full_subgraph(bigger_dag):
    result = bigger_dag[["truth", "prediction"]:"confusion_count"]
    expected = "DAG(func_nodes=[FuncNode(prediction,truth -> confusion_count_ -> confusion_count)], name=None)"
    assert result.__repr__() == expected
