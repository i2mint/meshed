import pytest

# just to shut the linter up about these
from i2 import Sig
from meshed.makers import code_to_dag


def user_story_01():
    # Sure, linter complains that names are not known, but all we want is valid code.
    # TODO: How to shut the linter up on this?

    # simple function calls
    data_source = get_data_source()  # no inputs
    wfs = make_wfs(data_source)  # one input
    chks = chunker(wfs, chk_size)  # two (positional) inputs

    # verify that we can handle multiple outputs (split)
    train_chks, test_chks = splitter(chks)

    # verify that we can handle k=v inputs (if v is a variable name):
    featurizer_obj = learn_featurizer(featurizer_learner, train_data=train_chks)


def test_user_story_01():
    dag = code_to_dag(user_story_01)
    assert (
        dag.synopsis_string()
        == ''' -> get_data_source -> data_source
data_source -> make_wfs -> wfs
wfs,chk_size -> chunker -> chks
chks -> splitter -> train_chks__test_chks
train_chks__test_chks -> test_chks__1 -> test_chks
train_chks__test_chks -> train_chks__0 -> train_chks
featurizer_learner,train_chks -> learn_featurizer -> featurizer_obj'''
    )
    assert str(Sig(dag)) == '(chk_size, featurizer_learner)'


def test_smoke_code_to_dag(src=user_story_01):
    dag = code_to_dag(src)


call, src_to_wf, data_src, chunker, chain, featurizer, model = [object] * 7


def user_story_02():
    wfs = call(src_to_wf, data_src)
    chks_iter = map(chunker, wfs)
    chks = chain(chks_iter)
    fvs = map(featurizer, chks)
    model_outputs = map(model, fvs)
