"""
seriously modified version of yahoo/graphkit
"""

# ---------- base --------------------------------------------------------------


class Data(object):
    """
    This wraps any data that is consumed or produced
    by a Operation. This data should also know how to serialize
    itself appropriately.
    This class an "abstract" class that should be extended by
    any class working with data in the HiC framework.
    """

    def __init__(self, **kwargs):
        pass

    def get_data(self):
        raise NotImplementedError

    def set_data(self, data):
        raise NotImplementedError


from dataclasses import dataclass, field


@dataclass
class Operation:
    """
    This is an abstract class representing a data transformation. To use this,
    please inherit from this class and customize the ``.compute`` method to your
    specific application.

    Names may be given to this layer and its inputs and outputs. This is
    important when connecting layers and data in a Network object, as the
    names are used to construct the graph.
    :param str name: The name the operation (e.g. conv1, conv2, etc..)
    :param list needs: Names of input data objects this layer requires.
    :param list provides: Names of output data objects this provides.
    :param dict params: A dict of key/value pairs representing parameters
                        associated with your operation. These values will be
                        accessible using the ``.params`` attribute of your object.
                        NOTE: It's important that any values stored in this
                        argument must be pickelable.
    """

    name: str = field(default='None')
    needs: list = field(default=None)
    provides: list = field(default=None)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        This method is a hook for you to override. It gets called after this
        object has been initialized with its ``needs``, ``provides``, ``name``,
        and ``params`` attributes. People often override this method to implement
        custom loading logic required for objects that do not pickle easily, and
        for initialization of c++ dependencies.
        """
        pass

    def __eq__(self, other):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return bool(self.name is not None and self.name == getattr(other, 'name', None))

    def __hash__(self):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return hash(self.name)

    def compute(self, inputs):
        """
        This method must be implemented to perform this layer's feed-forward
        computation on a given set of inputs.
        :param list inputs:
            A list of :class:`Data` objects on which to run the layer's
            feed-forward computation.
        :returns list:
            Should return a list of :class:`Data` objects representing
            the results of running the feed-forward computation on
            ``inputs``.
        """

        raise NotImplementedError

    def _compute(self, named_inputs, outputs=None):
        inputs = [named_inputs[d] for d in self.needs]
        results = self.compute(inputs)

        results = zip(self.provides, results)
        if outputs:
            outputs = set(outputs)
            results = filter(lambda x: x[0] in outputs, results)

        return dict(results)

    def __getstate__(self):
        """
        This allows your operation to be pickled.
        Everything needed to instantiate your operation should be defined by the
        following attributes: params, needs, provides, and name
        No other piece of state should leak outside of these 4 variables
        """

        result = {}
        # this check should get deprecated soon. its for downward compatibility
        # with earlier pickled operation objects
        if hasattr(self, 'params'):
            result['params'] = self.__dict__['params']
        result['needs'] = self.__dict__['needs']
        result['provides'] = self.__dict__['provides']
        result['name'] = self.__dict__['name']

        return result

    def __setstate__(self, state):
        """
        load from pickle and instantiate the detector
        """
        for k in iter(state):
            self.__setattr__(k, state[k])
        self.__postinit__()

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        return "%s(name='%s', needs=%s, provides=%s)" % (
            self.__class__.__name__,
            self.name,
            self.needs,
            self.provides,
        )


class NetworkOperation(Operation):
    def __init__(self, **kwargs):
        self.net = kwargs.pop('net')
        Operation.__init__(self, **kwargs)

        # set execution mode to single-threaded sequential by default
        self._execution_method = 'sequential'

    def _compute(self, named_inputs, outputs=None):
        return self.net.compute(outputs, named_inputs, method=self._execution_method)

    def __call__(self, *args, **kwargs):
        return self._compute(*args, **kwargs)

    def set_execution_method(self, method):
        """
        Determine how the network will be executed.
        Args:
            method: str
                If "parallel", execute graph operations concurrently
                using a threadpool.
        """
        options = ['parallel', 'sequential']
        assert method in options
        self._execution_method = method

    def plot(self, filename=None, show=False):
        self.net.plot(filename=filename, show=show)

    def __getstate__(self):
        state = Operation.__getstate__(self)
        state['net'] = self.__dict__['net']
        return state


# ------------ modifiers -------------------------------------------------------

"""
This sub-module contains input/output modifiers that can be applied to
arguments to ``needs`` and ``provides`` to let GraphKit know it should treat
them differently.

Copyright 2016, Yahoo Inc.
Licensed under the terms of the Apache License, Version 2.0. See the LICENSE
file associated with the project for terms.
"""


class optional(str):
    """
    Input values in ``needs`` may be designated as optional using this modifier.
    If this modifier is applied to an input value, that value will be input to
    the ``operation`` if it is available.  The function underlying the
    ``operation`` should have a parameter with the same name as the input value
    in ``needs``, and the input value will be passed as a keyword argument if
    it is available.

    Here is an example of an operation that uses an optional argument::

        from graphkit import operation, compose
        from graphkit.modifiers import optional

        # Function that adds either two or three numbers.
        def myadd(a, b, c=0):
            return a + b + c

        # Designate c as an optional argument.
        graph = compose('mygraph')(
            operator(name='myadd', needs=['a', 'b', optional('c')], provides='sum')(myadd)
        )

        # The graph works with and without 'c' provided as input.
        assert graph({'a': 5, 'b': 2, 'c': 4})['sum'] == 11
        assert graph({'a': 5, 'b': 2})['sum'] == 7

    """

    pass


# ------------ network ------------------------------------------------------

# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.


from contextlib import suppress

with suppress(ModuleNotFoundError, ImportError):
    import time
    import os
    import networkx as nx

    from io import StringIO

    # uses base.Operation

    class DataPlaceholderNode(str):
        """
        A node for the Network graph that describes the name of a Data instance
        produced or required by a layer.
        """

        def __repr__(self):
            return 'DataPlaceholderNode("%s")' % self

    class DeleteInstruction(str):
        """
        An instruction for the compiled list of evaluation steps to free or delete
        a Data instance from the Network's cache after it is no longer needed.
        """

        def __repr__(self):
            return 'DeleteInstruction("%s")' % self

    class Network(object):
        """
        This is the main network implementation. The class contains all of the
        code necessary to weave together operations into a directed-acyclic-graph (DAG)
        and pass data through.
        """

        def __init__(self, **kwargs):
            """ """

            # directed graph of layer instances and data-names defining the net.
            self.graph = nx.DiGraph()
            self._debug = kwargs.get('debug', False)

            # this holds the timing information for eache layer
            self.times = {}

            # a compiled list of steps to evaluate layers *in order* and free mem.
            self.steps = []

            # This holds a cache of results for the _find_necessary_steps
            # function, this helps speed up the compute call as well avoid
            # a multithreading issue that is occuring when accessing the
            # graph in networkx
            self._necessary_steps_cache = {}

        def add_op(self, operation):
            """
            Adds the given operation and its data requirements to the network graph
            based on the name of the operation, the names of the operation's needs, and
            the names of the data it provides.

            :param Operation operation: Operation object to add.
            """

            # assert layer and its data requirements are named.
            assert operation.name, 'Operation must be named'
            assert operation.needs is not None, "Operation's 'needs' must be named"
            assert (
                operation.provides is not None
            ), "Operation's 'provides' must be named"

            # assert layer is only added once to graph
            assert (
                operation not in self.graph.nodes()
            ), 'Operation may only be added once'

            # add nodes and edges to graph describing the data needs for this layer
            for n in operation.needs:
                self.graph.add_edge(DataPlaceholderNode(n), operation)

            # add nodes and edges to graph describing what this layer provides
            for p in operation.provides:
                self.graph.add_edge(operation, DataPlaceholderNode(p))

            # clear compiled steps (must recompile after adding new layers)
            self.steps = []

        def list_layers(self):
            assert self.steps, 'network must be compiled before listing layers.'
            return [(s.name, s) for s in self.steps if isinstance(s, Operation)]

        def show_layers(self):
            """Shows info (name, needs, and provides) about all layers in this network."""
            for name, step in self.list_layers():
                print('layer_name: ', name)
                print('\t', 'needs: ', step.needs)
                print('\t', 'provides: ', step.provides)
                print('')

        def compile(self):
            """Create a set of steps for evaluating layers
            and freeing memory as necessary"""

            # clear compiled steps
            self.steps = []

            # create an execution order such that each layer's needs are provided.
            ordered_nodes = list(nx.dag.topological_sort(self.graph))

            # add Operations evaluation steps, and instructions to free data.
            for i, node in enumerate(ordered_nodes):

                if isinstance(node, DataPlaceholderNode):
                    continue

                elif isinstance(node, Operation):

                    # add layer to list of steps
                    self.steps.append(node)

                    # Add instructions to delete predecessors as possible.  A
                    # predecessor may be deleted if it is a data placeholder that
                    # is no longer needed by future Operations.
                    for predecessor in self.graph.predecessors(node):
                        if self._debug:
                            print('checking if node %s can be deleted' % predecessor)
                        predecessor_still_needed = False
                        for future_node in ordered_nodes[i + 1 :]:
                            if isinstance(future_node, Operation):
                                if predecessor in future_node.needs:
                                    predecessor_still_needed = True
                                    break
                        if not predecessor_still_needed:
                            if self._debug:
                                print(
                                    '  adding delete instruction for %s' % predecessor
                                )
                            self.steps.append(DeleteInstruction(predecessor))

                else:
                    raise TypeError('Unrecognized network graph node')

        def _find_necessary_steps(self, outputs, inputs):
            """
            Determines what graph steps need to pe run to get to the requested
            outputs from the provided inputs.  Eliminates steps that come before
            (in topological order) any inputs that have been provided.  Also
            eliminates steps that are not on a path from he provided inputs to
            the requested outputs.

            :param list outputs:
                A list of desired output names.  This can also be ``None``, in which
                case the necessary steps are all graph nodes that are reachable
                from one of the provided inputs.

            :param dict inputs:
                A dictionary mapping names to values for all provided inputs.

            :returns:
                Returns a list of all the steps that need to be run for the
                provided inputs and requested outputs.
            """

            # return steps if it has already been computed before for this set of inputs and outputs
            outputs = (
                tuple(sorted(outputs)) if isinstance(outputs, (list, set)) else outputs
            )
            inputs_keys = tuple(sorted(inputs.keys()))
            cache_key = (inputs_keys, outputs)
            if cache_key in self._necessary_steps_cache:
                return self._necessary_steps_cache[cache_key]

            graph = self.graph
            if not outputs:

                # If caller requested all outputs, the necessary nodes are all
                # nodes that are reachable from one of the inputs.  Ignore input
                # names that aren't in the graph.
                necessary_nodes = set()
                for input_name in iter(inputs):
                    if graph.has_node(input_name):
                        necessary_nodes |= nx.descendants(graph, input_name)

            else:

                # If the caller requested a subset of outputs, find any nodes that
                # are made unecessary because we were provided with an input that's
                # deeper into the network graph.  Ignore input names that aren't
                # in the graph.
                unnecessary_nodes = set()
                for input_name in iter(inputs):
                    if graph.has_node(input_name):
                        unnecessary_nodes |= nx.ancestors(graph, input_name)

                # Find the nodes we need to be able to compute the requested
                # outputs.  Raise an exception if a requested output doesn't
                # exist in the graph.
                necessary_nodes = set()
                for output_name in outputs:
                    if not graph.has_node(output_name):
                        raise ValueError(
                            'graphkit graph does not have an output '
                            'node named %s' % output_name
                        )
                    necessary_nodes |= nx.ancestors(graph, output_name)

                # Get rid of the unnecessary nodes from the set of necessary ones.
                necessary_nodes -= unnecessary_nodes

            necessary_steps = [step for step in self.steps if step in necessary_nodes]

            # save this result in a precomputed cache for future lookup
            self._necessary_steps_cache[cache_key] = necessary_steps

            # Return an ordered list of the needed steps.
            return necessary_steps

        def compute(self, outputs, named_inputs, method=None):
            """
            Run the graph. Any inputs to the network must be passed in by name.

            :param list output: The names of the data node you'd like to have returned
                                once all necessary computations are complete.
                                If you set this variable to ``None``, all
                                data nodes will be kept and returned at runtime.

            :param dict named_inputs: A dict of key/value pairs where the keys
                                      represent the data nodes you want to populate,
                                      and the values are the concrete values you
                                      want to set for the data node.


            :returns: a dictionary of output data objects, keyed by name.
            """

            # assert that network has been compiled
            assert self.steps, 'network must be compiled before calling compute.'
            assert (
                isinstance(outputs, (list, tuple)) or outputs is None
            ), 'The outputs argument must be a list'

            # choose a method of execution
            if method == 'parallel':
                return self._compute_thread_pool_barrier_method(named_inputs, outputs)
            else:
                return self._compute_sequential_method(named_inputs, outputs)

        def _compute_thread_pool_barrier_method(
            self, named_inputs, outputs, thread_pool_size=10
        ):
            """
            This method runs the graph using a parallel pool of thread executors.
            You may achieve lower total latency if your graph is sufficiently
            sub divided into operations using this method.
            """
            from multiprocessing.dummy import Pool

            # if we have not already created a thread_pool, create one
            if not hasattr(self, '_thread_pool'):
                self._thread_pool = Pool(thread_pool_size)
            pool = self._thread_pool

            cache = {}
            cache.update(named_inputs)
            necessary_nodes = self._find_necessary_steps(outputs, named_inputs)

            # this keeps track of all nodes that have already executed
            has_executed = set()

            # with each loop iteration, we determine a set of operations that can be
            # scheduled, then schedule them onto a thread pool, then collect their
            # results onto a memory cache for use upon the next iteration.
            while True:

                # the upnext list contains a list of operations for scheduling
                # in the current round of scheduling
                upnext = []
                for node in necessary_nodes:
                    # only delete if all successors for the data node have been executed
                    if isinstance(node, DeleteInstruction):
                        if ready_to_delete_data_node(node, has_executed, self.graph):
                            if node in cache:
                                cache.pop(node)

                    # continue if this node is anything but an operation node
                    if not isinstance(node, Operation):
                        continue

                    if (
                        ready_to_schedule_operation(node, has_executed, self.graph)
                        and node not in has_executed
                    ):
                        upnext.append(node)

                # stop if no nodes left to schedule, exit out of the loop
                if len(upnext) == 0:
                    break

                done_iterator = pool.imap_unordered(
                    lambda op: (op, op._compute(cache)), upnext
                )
                for op, result in done_iterator:
                    cache.update(result)
                    has_executed.add(op)

            if not outputs:
                return cache
            else:
                return {k: cache[k] for k in iter(cache) if k in outputs}

        def _compute_sequential_method(self, named_inputs, outputs):
            """
            This method runs the graph one operation at a time in a single thread
            """
            # start with fresh data cache
            cache = {}

            # add inputs to data cache
            cache.update(named_inputs)

            # Find the subset of steps we need to run to get to the requested
            # outputs from the provided inputs.
            all_steps = self._find_necessary_steps(outputs, named_inputs)

            self.times = {}
            for step in all_steps:

                if isinstance(step, Operation):

                    if self._debug:
                        print('-' * 32)
                        print('executing step: %s' % step.name)

                    # time execution...
                    t0 = time.time()

                    # compute layer outputs
                    layer_outputs = step._compute(cache)

                    # add outputs to cache
                    cache.update(layer_outputs)

                    # record execution time
                    t_complete = round(time.time() - t0, 5)
                    self.times[step.name] = t_complete
                    if self._debug:
                        print('step completion time: %s' % t_complete)

                # Process DeleteInstructions by deleting the corresponding data
                # if possible.
                elif isinstance(step, DeleteInstruction):

                    if outputs and step not in outputs:
                        # Some DeleteInstruction steps may not exist in the cache
                        # if they come from optional() needs that are not privoded
                        # as inputs.  Make sure the step exists before deleting.
                        if step in cache:
                            if self._debug:
                                print("removing data '%s' from cache." % step)
                            cache.pop(step)

                else:
                    raise TypeError('Unrecognized instruction.')

            if not outputs:
                # Return the whole cache as output, including input and
                # intermediate data nodes.
                return cache

            else:
                # Filter outputs to just return what's needed.
                # Note: list comprehensions exist in python 2.7+
                return {k: cache[k] for k in iter(cache) if k in outputs}

        def plot(self, filename=None, show=False):
            """
            Plot the graph.

            params:
            :param str filename:
                Write the output to a png, pdf, or graphviz dot file. The extension
                controls the output format.

            :param boolean show:
                If this is set to True, use matplotlib to show the graph diagram
                (Default: False)

            :returns:
                An instance of the pydot graph

            """
            from contextlib import suppress

            with suppress(ModuleNotFoundError, ImportError):
                import pydot
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg

                assert self.graph is not None

                def get_node_name(a):
                    if isinstance(a, DataPlaceholderNode):
                        return a
                    return a.name

                g = pydot.Dot(graph_type='digraph')

                # draw nodes
                for nx_node in self.graph.nodes():
                    if isinstance(nx_node, DataPlaceholderNode):
                        node = pydot.Node(name=nx_node, shape='rect')
                    else:
                        node = pydot.Node(name=nx_node.name, shape='circle')
                    g.add_node(node)

                # draw edges
                for src, dst in self.graph.edges():
                    src_name = get_node_name(src)
                    dst_name = get_node_name(dst)
                    edge = pydot.Edge(src=src_name, dst=dst_name)
                    g.add_edge(edge)

                # save plot
                if filename:
                    basename, ext = os.path.splitext(filename)
                    with open(filename, 'w') as fh:
                        if ext.lower() == '.png':
                            fh.write(g.create_png())
                        elif ext.lower() == '.dot':
                            fh.write(g.to_string())
                        elif ext.lower() in ['.jpg', '.jpeg']:
                            fh.write(g.create_jpeg())
                        elif ext.lower() == '.pdf':
                            fh.write(g.create_pdf())
                        elif ext.lower() == '.svg':
                            fh.write(g.create_svg())
                        else:
                            raise Exception(
                                'Unknown file format for saving graph: %s' % ext
                            )

                # display graph via matplotlib
                if show:
                    png = g.create_png()
                    sio = StringIO(png)
                    img = mpimg.imread(sio)
                    plt.imshow(img, aspect='equal')
                    plt.show()

                return g

    def ready_to_schedule_operation(op, has_executed, graph):
        """
        Determines if a Operation is ready to be scheduled for execution based on
        what has already been executed.

        Args:
            op:
                The Operation object to check
            has_executed: set
                A set containing all operations that have been executed so far
            graph:
                The networkx graph containing the operations and data nodes
        Returns:
            A boolean indicating whether the operation may be scheduled for
            execution based on what has already been executed.
        """
        dependencies = set(
            filter(lambda v: isinstance(v, Operation), nx.ancestors(graph, op))
        )
        return dependencies.issubset(has_executed)

    def ready_to_delete_data_node(name, has_executed, graph):
        """
        Determines if a DataPlaceholderNode is ready to be deleted from the
        cache.

        Args:
            name:
                The name of the data node to check
            has_executed: set
                A set containing all operations that have been executed so far
            graph:
                The networkx graph containing the operations and data nodes
        Returns:
            A boolean indicating whether the data node can be deleted or not.
        """
        data_node = get_data_node(name, graph)
        return set(graph.successors(data_node)).issubset(has_executed)

    def get_data_node(name, graph):
        """
        Gets a data node from a graph using its name
        """
        for node in graph.nodes():
            if node == name and isinstance(node, DataPlaceholderNode):
                return node
        return None

    # ------------ functional ------------------------------------------------------

    # Copyright 2016, Yahoo Inc.
    # Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

    from itertools import chain

    # uses Operation, NetworkOperation from base
    # uses Network from network

    class FunctionalOperation(Operation):
        def __init__(self, **kwargs):
            self.fn = kwargs.pop('fn')
            Operation.__init__(self, **kwargs)

        def _compute(self, named_inputs, outputs=None):
            inputs = [
                named_inputs[d] for d in self.needs if not isinstance(d, optional)
            ]

            # Find any optional inputs in named_inputs.  Get only the ones that
            # are present there, no extra `None`s.
            optionals = {
                n: named_inputs[n]
                for n in self.needs
                if isinstance(n, optional) and n in named_inputs
            }

            # Combine params and optionals into one big glob of keyword arguments.
            kwargs = {k: v for d in (self.params, optionals) for k, v in d.items()}
            result = self.fn(*inputs, **kwargs)
            if len(self.provides) == 1:
                result = [result]

            result = zip(self.provides, result)
            if outputs:
                outputs = set(outputs)
                result = filter(lambda x: x[0] in outputs, result)

            return dict(result)

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def __getstate__(self):
            state = Operation.__getstate__(self)
            state['fn'] = self.__dict__['fn']
            return state

    class operation(Operation):
        """
        This object represents an operation in a computation graph.  Its
        relationship to other operations in the graph is specified via its
        ``needs`` and ``provides`` arguments.

        :param function fn:
            The function used by this operation.  This does not need to be
            specified when the operation object is instantiated and can instead
            be set via ``__call__`` later.

        :param str name:
            The name of the operation in the computation graph.

        :param list needs:
            Names of input data objects this operation requires.  These should
            correspond to the ``args`` of ``fn``.

        :param list provides:
            Names of output data objects this operation provides.

        :param dict params:
            A dict of key/value pairs representing constant parameters
            associated with your operation.  These can correspond to either
            ``args`` or ``kwargs`` of ``fn`.
        """

        def __init__(self, fn=None, **kwargs):
            self.fn = fn
            Operation.__init__(self, **kwargs)

        def _normalize_kwargs(self, kwargs):

            # Allow single value for needs parameter
            if 'needs' in kwargs and type(kwargs['needs']) == str:
                assert kwargs['needs'], 'empty string provided for `needs` parameters'
                kwargs['needs'] = [kwargs['needs']]

            # Allow single value for provides parameter
            if 'provides' in kwargs and type(kwargs['provides']) == str:
                assert kwargs[
                    'provides'
                ], 'empty string provided for `needs` parameters'
                kwargs['provides'] = [kwargs['provides']]

            assert kwargs['name'], 'operation needs a name'
            assert type(kwargs['needs']) == list, 'no `needs` parameter provided'
            assert type(kwargs['provides']) == list, 'no `provides` parameter provided'
            assert hasattr(
                kwargs['fn'], '__call__'
            ), 'operation was not provided with a callable'

            if type(kwargs['params']) is not dict:
                kwargs['params'] = {}

            return kwargs

        def __call__(self, fn=None, **kwargs):
            """
            This enables ``operation`` to act as a decorator or as a functional
            operation, for example::

                @operator(name='myadd1', needs=['a', 'b'], provides=['c'])
                def myadd(a, b):
                    return a + b

            or::

                def myadd(a, b):
                    return a + b
                operator(name='myadd1', needs=['a', 'b'], provides=['c'])(myadd)

            :param function fn:
                The function to be used by this ``operation``.

            :return:
                Returns an operation class that can be called as a function or
                composed into a computation graph.
            """

            if fn is not None:
                self.fn = fn

            total_kwargs = {}
            total_kwargs.update(vars(self))
            total_kwargs.update(kwargs)
            total_kwargs = self._normalize_kwargs(total_kwargs)

            return FunctionalOperation(**total_kwargs)

        def __repr__(self):
            """
            Display more informative names for the Operation class
            """
            return "%s(name='%s', needs=%s, provides=%s, fn=%s)" % (
                self.__class__.__name__,
                self.name,
                self.needs,
                self.provides,
                self.fn.__name__,
            )

    class compose(object):
        """
        This is a simple class that's used to compose ``operation`` instances into
        a computation graph.

        :param str name:
            A name for the graph being composed by this object.

        :param bool merge:
            If ``True``, this compose object will attempt to merge together
            ``operation`` instances that represent entire computation graphs.
            Specifically, if one of the ``operation`` instances passed to this
            ``compose`` object is itself a graph operation created by an
            earlier use of ``compose`` the sub-operations in that graph are
            compared against other operations passed to this ``compose``
            instance (as well as the sub-operations of other graphs passed to
            this ``compose`` instance).  If any two operations are the same
            (based on name), then that operation is computed only once, instead
            of multiple times (one for each time the operation appears).
        """

        def __init__(self, name=None, merge=False):
            assert name, 'compose needs a name'
            self.name = name
            self.merge = merge

        def __call__(self, *operations):
            """
            Composes a collection of operations into a single computation graph,
            obeying the ``merge`` property, if set in the constructor.

            :param operations:
                Each argument should be an operation instance created using
                ``operation``.

            :return:
                Returns a special type of operation class, which represents an
                entire computation graph as a single operation.
            """
            assert len(operations), 'no operations provided to compose'

            # If merge is desired, deduplicate operations before building network
            if self.merge:
                merge_set = set()
                for op in operations:
                    if isinstance(op, NetworkOperation):
                        net_ops = filter(
                            lambda x: isinstance(x, Operation), op.net.steps
                        )
                        merge_set.update(net_ops)
                    else:
                        merge_set.add(op)
                operations = list(merge_set)

            def order_preserving_uniquifier(seq, seen=None):
                seen = seen if seen else set()
                seen_add = seen.add
                return [x for x in seq if not (x in seen or seen_add(x))]

            provides = order_preserving_uniquifier(
                chain(*[op.provides for op in operations])
            )
            needs = order_preserving_uniquifier(
                chain(*[op.needs for op in operations]), set(provides)
            )

            # compile network
            net = Network()
            for op in operations:
                net.add_op(op)
            net.compile()

            return NetworkOperation(
                name=self.name, needs=needs, provides=provides, params={}, net=net
            )
