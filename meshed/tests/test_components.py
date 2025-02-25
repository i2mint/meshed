"""Tests for components.py"""

from meshed.components import Itemgetter, AttrGetter


def test_extractors():
    def data1():
        return {0: "start", 1: "stop", "bob": "alice"}

    first_extraction = Itemgetter([0, 1], name="first_extraction", input_name="data1")
    second_extraction = Itemgetter("bob", name="second_extraction", input_name="data1")

    from meshed import DAG

    dag1 = DAG([data1, first_extraction, second_extraction])

    dag1.synopsis_string() == """ -> data1_ -> data1
data -> first_extraction_ -> first_extraction
data -> second_extraction_ -> second_extraction"""

    assert dag1() == (("start", "stop"), "alice")

    def data2():
        from collections import namedtuple

        return namedtuple("data", ["one", "two", "bob"])("start", "stop", "alice")

    # Note: data produces a namedtuple looking like this
    d = data2()
    assert (d.one, d.two, d.bob) == ("start", "stop", "alice")

    extractor_1 = AttrGetter(["one", "two"], name="one_and_two", input_name="data2")
    extractor_2 = AttrGetter("bob", name="bob", input_name="data2")

    dag2 = DAG([data2, extractor_1, extractor_2])

    assert dag2() == (("start", "stop"), "alice")

    dag2.synopsis_string() == """ -> data2_ -> data2
data -> one_and_two_ -> one_and_two
data -> bob_ -> bob"""
