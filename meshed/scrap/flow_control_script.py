from meshed import DAG
from creek.automatas import BasicAutomata, mapping_to_transition_func
from typing import Callable, MutableMapping, Any, Mapping, Literal
from dataclasses import dataclass
from i2 import ch_names


Case = Any
Cases = Mapping[Case, Callable]


RecordingCommands = Literal["start", "resume", "stop"]


def mk_test_objects():
    # from slang import fixed_step_chunker

    audio = range(100)
    audio_chk_size = 5
    # audio_chks = list(fixed_step_chunker(audio, chk_size=audio_chk_size))
    audio_chks = [
        audio[i : i + audio_chk_size] for i in range(0, len(audio), audio_chk_size)
    ]
    plc_values = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]

    return audio_chks, plc_values


@dataclass
class RecordingSwitchBoard:
    store: MutableMapping = None
    _current_key = None

    def start(self, key, chk):
        self._current_key = key
        self.store[key] = []
        self._append(chk)

    def resume(self, key, chk):
        print(f"resume called")
        self._append(chk)

    def stop(self, key, chk):
        self._append(chk)
        self._current_key = None

    def _append(self, chk):
        if self._current_key is None:
            raise ValueError("Cannot append without first starting recording.")
        self.store[self._current_key].extend(chk)

    @property
    def is_recording(self):
        return self._current_key is not None


@dataclass
class SimpleSwitchCase:
    """A functional implementation of thw switch-case control flow.
    Makes a callable that takes two arguments, a case and an input.

    >>> f = SimpleSwitchCase({'plus_one': lambda x: x + 1, 'times_two': lambda x: x * 2})
    >>> f('plus_one', 2)
    3
    >>> f('times_two', 2)
    4
    """

    cases: Mapping[Case, Callable]

    def __call__(self, case, input):
        func = self.cases.get(case, None)
        if func is None:
            raise ValueError(f"Case {case} not found.")
        return func(input)


def mk_simple_switch_case(
    cases: Cases, *, name: str = None, case_name: str = None, input_name: str = None
):
    """
    Makes a simple switch-case function, with optional naming control.
    """
    switch_case_func = SimpleSwitchCase(cases)
    switch_case_func = ch_names(
        switch_case_func, **dict(case=case_name, input=input_name)
    )
    if name is not None:
        switch_case_func.__name__ = name
    return switch_case_func


def mk_recorder_switch(
    store, *, mk_recorder: Callable[[MutableMapping], Any] = RecordingSwitchBoard
):
    recorder = mk_recorder(store)
    return mk_simple_switch_case(
        {
            "start": lambda key_and_chk: recorder.start(*key_and_chk),
            "resume": lambda key_and_chk: recorder.resume(*key_and_chk),
            "stop": lambda key_and_chk: recorder.stop(*key_and_chk),
            "waiting": lambda x: None,
        },
        name="recorder_switch",
        case_name="state",
        input_name="key_and_chk",
    )


def mk_transition_func(
    trans_func_mapping,
    initial_state,  # symbol_var_name: str,
):
    recording_state_transition_func = mapping_to_transition_func(
        trans_func_mapping,
        strict=False,
    )
    transitioner = BasicAutomata(
        transition_func=recording_state_transition_func,
        state=initial_state,
    )

    # @i2.ch_names(symbol=symbol_var_name)
    def transition(symbol):
        return transitioner.transition(symbol)

    # transition = transitioner.reset().transition

    return transition


# store = mk_recorder_switch(store)
trans_func_mapping = {
    ("waiting", 1): "start",
    ("start", 0): "resume",
    ("start", 1): "stop",
    ("resume", 1): "stop",
    ("stop", 0): "waiting",
    ("stop", 1): "start",
}

# debugging tools
logger = {
    "symbol": [],
    "state": [],
    "state_func": [],
    "transition_func": [],
    "recorder": [],
}

# TFunc = mk_transition_func(trans_func_mapping, "waiting")
dag = DAG.from_funcs(
    recorder_switch=lambda store: mk_recorder_switch(store),
    recorder_logger=lambda recorder_switch: logger["recorder"].append(
        id(recorder_switch)
    ),
    # debug = lambda recorder_switch: print(id(recorder_switch)),
    transition_func=lambda trans_func_mapping: mk_transition_func(
        trans_func_mapping, "waiting"
    ),
    transition_logger=lambda transition_func: logger["transition_func"].append(
        transition_func
    ),
    symbol=lambda plc: plc,
    symbol_logger=lambda symbol: logger["symbol"].append(symbol),
    state=lambda transition_func, symbol: transition_func(symbol),
    # tFunc=lambda: TFunc,
    # state=lambda tFunc, symbol: tFunc(symbol),
    state_logger=lambda state: logger["state"].append(state),
    key_and_chk=lambda key, chk: (key, chk),
    # key_and_chk_logger=lambda key_and_chk: logger['key_and_chk'].append(key_and_chk),
    state_func=lambda recorder_switch, state, key_and_chk: recorder_switch(
        state, key_and_chk
    ),
    state_func_logger=lambda state_func: logger["state_func"].append(state_func),
    output=lambda state_func, key_and_chk: (
        state_func(*key_and_chk) if state_func is not None else None
    ),
    result=lambda recorder_switch, transition_func, symbol, state, key_and_chk, state_func, output: dict(
        recorder_switch=recorder_switch,
        transition_func=transition_func,
        symbol=symbol,
        state=state,
        key_and_chk=key_and_chk,
        state_func=state_func,
        output=output,
    ),
)


if __name__ == "__main__":
    store = dict()

    my_dag = dag.partial(store=store, trans_func_mapping=trans_func_mapping)
    # my_dag.dot_digraph()
    # print(i2.Sig(my_dag))

    audio_chks, plc_values = mk_test_objects()
    keys = range(max(len(audio_chks), len(plc_values)))  # need some source of keys now!

    for chk, plc, key in zip(audio_chks, plc_values, keys):
        # print(f"{store =}{chk=} {plc=} {key=} ")
        # print(f'{store=}')
        # print(f"{my_dag.last_scope=}")

        # print(f"{my_dag[:'state_func'](chk=chk, plc=plc, key=key)}")
        # res = my_dag(chk=chk, plc=plc, key=key)
        res = dag(
            store=store,
            trans_func_mapping=trans_func_mapping,
            chk=chk,
            plc=plc,
            key=key,
        )

        # print(store)  # Careful: use keyword
    print(logger["symbol"])

    print(logger["state"])
    print(logger["recorder"])

    # transitioner = logger['transition_func'][-1]
    transitioner = TFunc
    state_sequence = list(map(transitioner, plc_values))
    print(state_sequence)
    # print(transitioner)
