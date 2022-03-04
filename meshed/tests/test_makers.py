import pytest

# just to shut the linter up about these
call, src_to_wf, data_src, chunker, chain, featurizer, model = [object] * 7


def user_story_01():
    wfs = call(src_to_wf, data_src)
    chks_iter = map(chunker, wfs)
    chks = chain(chks_iter)
    fvs = map(featurizer, chks)
    model_outputs = map(model, fvs)
