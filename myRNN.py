from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

import tensorflow as tf
# pylint: disable=protected-access
_state_size_with_prefix = rnn_cell._state_size_with_prefix
# pylint: enable=protected-access

def raw_rnn(cell, loop_fn,
            parallel_iterations=None, swap_memory=False, scope=None):
    """Creates an `RNN` specified by RNNCell `cell` and loop function `loop_fn`.

    **NOTE: This method is still in testing, and the API may change.**

    This function is a more primitive version of `dynamic_rnn` that provides
    more direct access to the inputs each iteration.  It also provides more
    control over when to start and finish reading the sequence, and
    what to emit for the output.

    For example, it can be used to implement the dynamic decoder of a seq2seq
    model.

    Instead of working with `Tensor` objects, most operations work with
    `TensorArray` objects directly.

    The operation of `raw_rnn`, in pseudo-code, is basically the following:

    ```
    time = tf.constant(0, dtype=tf.int32)
    (finished, next_input, initial_state, _, loop_state) = loop_fn(
        time=time, cell_output=None, cell_state=None, loop_state=None)
    emit_ta = TensorArray(dynamic_size=True, dtype=initial_state.dtype)
    state = initial_state
    while not all(finished):
      (output, cell_state) = cell(next_input, state)
      (next_finished, next_input, next_state, emit, loop_state) = loop_fn(
          time=time + 1, cell_output=output, cell_state=cell_state,
          loop_state=loop_state)
      # Emit zeros and copy forward state for minibatch entries that are finished.
      state = tf.select(finished, state, next_state)
      emit = tf.select(finished, tf.zeros_like(emit), emit)
      emit_ta = emit_ta.write(time, emit)
      # If any new minibatch entries are marked as finished, mark these
      finished = tf.logical_or(finished, next_finished)
      time += 1
    return (emit_ta, state, loop_state)
    ```

    with the additional properties that output and state may be (possibly nested)
    tuples, as determined by `cell.output_size` and `cell.state_size`, and
    as a result the final `state` and `emit_ta` may themselves be tuples.

    A simple implementation of `dynamic_rnn` via `raw_rnn` looks like this:

    ```python
    inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                            dtype=tf.float32)
    sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unpack(inputs)

    cell = tf.nn.rnn_cell.LSTMCell(num_units)

    def loop_fn(time, cell_output, cell_state, loop_state):
      emit_output = cell_output  # == None for time == 0
      if cell_output is None:  # time == 0
        next_cell_state = cell.zero_state(batch_size, tf.float32)
      else:
        next_cell_state = cell_state
      elements_finished = (time >= sequence_length)
      finished = tf.reduce_all(elements_finished)
      next_input = tf.cond(
          finished,
          lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
          lambda: inputs_ta.read(time))
      next_loop_state = None
      return (elements_finished, next_input, next_cell_state,
              emit_output, next_loop_state)

    outputs_ta, final_state, _ = raw_rnn(cell, loop_fn)
    outputs = outputs_ta.pack()
    ```

    Args:
      cell: An instance of RNNCell.
      loop_fn: A callable that takes inputs
        `(time, cell_output, cell_state, loop_state)`
        and returns the tuple
        `(finished, next_input, next_cell_state, emit_output, next_loop_state)`.
        Here `time` is an int32 scalar `Tensor`, `cell_output` is a
        `Tensor` or (possibly nested) tuple of tensors as determined by
        `cell.output_size`, and `cell_state` is a `Tensor`
        or (possibly nested) tuple of tensors, as determined by the `loop_fn`
        on its first call (and should match `cell.state_size`).
        The outputs are: `finished`, a boolean `Tensor` of
        shape `[batch_size]`, `next_input`: the next input to feed to `cell`,
        `next_cell_state`: the next state to feed to `cell`,
        and `emit_output`: the output to store for this iteration.

        Note that `emit_output` should be a `Tensor` or (possibly nested)
        tuple of tensors with shapes and structure matching `cell.output_size`
        and `cell_output` above.  The parameter `cell_state` and output
        `next_cell_state` may be either a single or (possibly nested) tuple
        of tensors.  The parameter `loop_state` and
        output `next_loop_state` may be either a single or (possibly nested) tuple
        of `Tensor` and `TensorArray` objects.  This last parameter
        may be ignored by `loop_fn` and the return value may be `None`.  If it
        is not `None`, then the `loop_state` will be propagated through the RNN
        loop, for use purely by `loop_fn` to keep track of its own state.
        The `next_loop_state` parameter returned may be `None`.

        The first call to `loop_fn` will be `time = 0`, `cell_output = None`,
        `cell_state = None`, and `loop_state = None`.  For this call:
        The `next_cell_state` value should be the value with which to initialize
        the cell's state.  It may be a final state from a previous RNN or it
        may be the output of `cell.zero_state()`.  It should be a
        (possibly nested) tuple structure of tensors.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a `TensorShape`, this must be a `Tensor` of
        appropriate type and shape `[batch_size] + cell.state_size`.
        If `cell.state_size` is a (possibly nested) tuple of ints or
        `TensorShape`, this will be a tuple having the corresponding shapes.
        The `emit_output` value may be  either `None` or a (possibly nested)
        tuple structure of tensors, e.g.,
        `(tf.zeros(shape_0, dtype=dtype_0), tf.zeros(shape_1, dtype=dtype_1))`.
        If this first `emit_output` return value is `None`,
        then the `emit_ta` result of `raw_rnn` will have the same structure and
        dtypes as `cell.output_size`.  Otherwise `emit_ta` will have the same
        structure, shapes (prepended with a `batch_size` dimension), and dtypes
        as `emit_output`.  The actual values returned for `emit_output` at this
        initializing call are ignored.  Note, this emit structure must be
        consistent across all time steps.

      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
      A tuple `(emit_ta, final_state, final_loop_state)` where:

      `emit_ta`: The RNN output `TensorArray`.
         If `loop_fn` returns a (possibly nested) set of Tensors for
         `emit_output` during initialization, (inputs `time = 0`,
         `cell_output = None`, and `loop_state = None`), then `emit_ta` will
         have the same structure, dtypes, and shapes as `emit_output` instead.
         If `loop_fn` returns `emit_output = None` during this call,
         the structure of `cell.output_size` is used:
         If `cell.output_size` is a (possibly nested) tuple of integers
         or `TensorShape` objects, then `emit_ta` will be a tuple having the
         same structure as `cell.output_size`, containing TensorArrays whose
         elements' shapes correspond to the shape data in `cell.output_size`.

      `final_state`: The final cell state.  If `cell.state_size` is an int, this
        will be shaped `[batch_size, cell.state_size]`.  If it is a
        `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
        If it is a (possibly nested) tuple of ints or `TensorShape`, this will
        be a tuple having the corresponding shapes.

      `final_loop_state`: The final loop state as returned by `loop_fn`.

    Raises:
      TypeError: If `cell` is not an instance of RNNCell, or `loop_fn` is not
        a `callable`.
    """

    if not isinstance(cell, rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not callable(loop_fn):
        raise TypeError("loop_fn must be a callable")

    parallel_iterations = parallel_iterations or 32

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        time = constant_op.constant(0, dtype=dtypes.int32)
        (elements_finished, next_input, initial_state, emit_structure,
         init_loop_state) = loop_fn(
             time, None, None, None, None)  # time, cell_output, cell_state, loop_state, emit_ta
        flat_input = nest.flatten(next_input)

        # Need a surrogate loop state for the while_loop if none is available.
        loop_state = (init_loop_state if init_loop_state is not None
                      else constant_op.constant(0, dtype=dtypes.int32))

        input_shape = [input_.get_shape() for input_ in flat_input]
        static_batch_size = input_shape[0][0]

        for input_shape_i in input_shape:
            # Static verification that batch sizes all match
            static_batch_size.merge_with(input_shape_i[0])

        batch_size = static_batch_size.value
        if batch_size is None:
            batch_size = array_ops.shape(flat_input[0])[0]

        nest.assert_same_structure(initial_state, cell.state_size)
        state = initial_state
        flat_state = nest.flatten(state)
        flat_state = [ops.convert_to_tensor(s) for s in flat_state]
        state = nest.pack_sequence_as(structure=state,
                                      flat_sequence=flat_state)

        if emit_structure is not None:
            flat_emit_structure = nest.flatten(emit_structure)
            flat_emit_size = [array_ops.shape(emit) for emit in flat_emit_structure]
            flat_emit_dtypes = [emit.dtype for emit in flat_emit_structure]
        else:
            raise ValueError('emit_structure is None')
        
        flat_emit_ta = [
            tensor_array_ops.TensorArray(
                dtype=dtype_i, dynamic_size=True, clear_after_read=False, size=0, name="rnn_output_%d" % i)
            for i, dtype_i in enumerate(flat_emit_dtypes)]
        emit_ta = nest.pack_sequence_as(structure=emit_structure,
                                        flat_sequence=flat_emit_ta)

        flat_zero_emit = [
            array_ops.zeros(
                size_i,
                dtype_i)
            for size_i, dtype_i in zip(flat_emit_size, flat_emit_dtypes)]
        
        zero_emit = nest.pack_sequence_as(structure=emit_structure,
                                          flat_sequence=flat_zero_emit)

        def condition(unused_time, elements_finished, *_):
            return math_ops.logical_not(math_ops.reduce_all(elements_finished))

        def body(time, elements_finished, current_input,
                 emit_ta, state, loop_state):
            """Internal while loop body for raw_rnn.

            Args:
              time: time scalar.
              elements_finished: batch-size vector.
              current_input: possibly nested tuple of input tensors.
              emit_ta: possibly nested tuple of output TensorArrays.
              state: possibly nested tuple of state tensors.
              loop_state: possibly nested tuple of loop state tensors.

            Returns:
              Tuple having the same size as Args but with updated values.
            """

            (next_output, cell_state) = cell(current_input, state)

            nest.assert_same_structure(state, cell_state)
            nest.assert_same_structure(cell.output_size, next_output)

            next_time = time + 1
            (next_finished, next_input, next_state, emit_output,
             next_loop_state) = loop_fn(
                 next_time, next_output, cell_state, loop_state, emit_ta)

            nest.assert_same_structure(state, next_state)
            nest.assert_same_structure(current_input, next_input)
            nest.assert_same_structure(emit_ta, emit_output)

            # If loop_fn returns None for next_loop_state, just reuse the
            # previous one.
            loop_state = loop_state if next_loop_state is None else next_loop_state

            def _copy_some_through(current, candidate):
                current_flat = nest.flatten(current)
                candidate_flat = nest.flatten(candidate)
                result_flat = [
                    math_ops.select(elements_finished, current_i, candidate_i)
                    for (current_i, candidate_i) in zip(current_flat, candidate_flat)]
                return nest.pack_sequence_as(
                    structure=current, flat_sequence=result_flat)

            emit_output = _copy_some_through(zero_emit, emit_output)
            next_state = _copy_some_through(state, next_state)

            emit_output_flat = nest.flatten(emit_output)
            emit_ta_flat = nest.flatten(emit_ta)

            elements_finished = math_ops.logical_or(elements_finished, next_finished)
                      
            emit_ta_flat = [
                ta.write(time, emit)
                for (ta, emit) in zip(emit_ta_flat, emit_output_flat)]

            emit_ta = nest.pack_sequence_as(
                structure=emit_structure, flat_sequence=emit_ta_flat)

            return (next_time, elements_finished, next_input,
                    emit_ta, next_state, loop_state)

        returned = control_flow_ops.while_loop(
            condition, body, loop_vars=[
                time, elements_finished, next_input,
                emit_ta, state, loop_state],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        (emit_ta, final_state, final_loop_state) = returned[-3:]

        if init_loop_state is None:
            final_loop_state = None
        
        return (emit_ta, final_state, final_loop_state)

def beam_search_rnn(cell, loop_fn,
            parallel_iterations=None, swap_memory=False, scope=None):
    """Creates an `RNN` specified by RNNCell `cell` and loop function `loop_fn`.

    **NOTE: This method is still in testing, and the API may change.**

    This function is a more primitive version of `dynamic_rnn` that provides
    more direct access to the inputs each iteration.  It also provides more
    control over when to start and finish reading the sequence, and
    what to emit for the output.

    For example, it can be used to implement the dynamic decoder of a seq2seq
    model.

    Instead of working with `Tensor` objects, most operations work with
    `TensorArray` objects directly.

    The operation of `raw_rnn`, in pseudo-code, is basically the following:

    ```
    time = tf.constant(0, dtype=tf.int32)
    (finished, next_input, initial_state, _, loop_state) = loop_fn(
        time=time, cell_output=None, cell_state=None, loop_state=None)
    emit_ta = TensorArray(dynamic_size=True, dtype=initial_state.dtype)
    state = initial_state
    while not all(finished):
      (output, cell_state) = cell(next_input, state)
      (next_finished, next_input, next_state, emit, loop_state) = loop_fn(
          time=time + 1, cell_output=output, cell_state=cell_state,
          loop_state=loop_state)
      # Emit zeros and copy forward state for minibatch entries that are finished.
      state = tf.select(finished, state, next_state)
      emit = tf.select(finished, tf.zeros_like(emit), emit)
      emit_ta = emit_ta.write(time, emit)
      # If any new minibatch entries are marked as finished, mark these
      finished = tf.logical_or(finished, next_finished)
      time += 1
    return (emit_ta, state, loop_state)
    ```

    with the additional properties that output and state may be (possibly nested)
    tuples, as determined by `cell.output_size` and `cell.state_size`, and
    as a result the final `state` and `emit_ta` may themselves be tuples.

    A simple implementation of `dynamic_rnn` via `raw_rnn` looks like this:

    ```python
    inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                            dtype=tf.float32)
    sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unpack(inputs)

    cell = tf.nn.rnn_cell.LSTMCell(num_units)

    def loop_fn(time, cell_output, cell_state, loop_state):
      emit_output = cell_output  # == None for time == 0
      if cell_output is None:  # time == 0
        next_cell_state = cell.zero_state(batch_size, tf.float32)
      else:
        next_cell_state = cell_state
      elements_finished = (time >= sequence_length)
      finished = tf.reduce_all(elements_finished)
      next_input = tf.cond(
          finished,
          lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
          lambda: inputs_ta.read(time))
      next_loop_state = None
      return (elements_finished, next_input, next_cell_state,
              emit_output, next_loop_state)

    outputs_ta, final_state, _ = raw_rnn(cell, loop_fn)
    outputs = outputs_ta.pack()
    ```

    Args:
      cell: An instance of RNNCell.
      loop_fn: A callable that takes inputs
        `(time, cell_output, cell_state, loop_state)`
        and returns the tuple
        `(finished, next_input, next_cell_state, emit_output, next_loop_state)`.
        Here `time` is an int32 scalar `Tensor`, `cell_output` is a
        `Tensor` or (possibly nested) tuple of tensors as determined by
        `cell.output_size`, and `cell_state` is a `Tensor`
        or (possibly nested) tuple of tensors, as determined by the `loop_fn`
        on its first call (and should match `cell.state_size`).
        The outputs are: `finished`, a boolean `Tensor` of
        shape `[batch_size]`, `next_input`: the next input to feed to `cell`,
        `next_cell_state`: the next state to feed to `cell`,
        and `emit_output`: the output to store for this iteration.

        Note that `emit_output` should be a `Tensor` or (possibly nested)
        tuple of tensors with shapes and structure matching `cell.output_size`
        and `cell_output` above.  The parameter `cell_state` and output
        `next_cell_state` may be either a single or (possibly nested) tuple
        of tensors.  The parameter `loop_state` and
        output `next_loop_state` may be either a single or (possibly nested) tuple
        of `Tensor` and `TensorArray` objects.  This last parameter
        may be ignored by `loop_fn` and the return value may be `None`.  If it
        is not `None`, then the `loop_state` will be propagated through the RNN
        loop, for use purely by `loop_fn` to keep track of its own state.
        The `next_loop_state` parameter returned may be `None`.

        The first call to `loop_fn` will be `time = 0`, `cell_output = None`,
        `cell_state = None`, and `loop_state = None`.  For this call:
        The `next_cell_state` value should be the value with which to initialize
        the cell's state.  It may be a final state from a previous RNN or it
        may be the output of `cell.zero_state()`.  It should be a
        (possibly nested) tuple structure of tensors.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a `TensorShape`, this must be a `Tensor` of
        appropriate type and shape `[batch_size] + cell.state_size`.
        If `cell.state_size` is a (possibly nested) tuple of ints or
        `TensorShape`, this will be a tuple having the corresponding shapes.
        The `emit_output` value may be  either `None` or a (possibly nested)
        tuple structure of tensors, e.g.,
        `(tf.zeros(shape_0, dtype=dtype_0), tf.zeros(shape_1, dtype=dtype_1))`.
        If this first `emit_output` return value is `None`,
        then the `emit_ta` result of `raw_rnn` will have the same structure and
        dtypes as `cell.output_size`.  Otherwise `emit_ta` will have the same
        structure, shapes (prepended with a `batch_size` dimension), and dtypes
        as `emit_output`.  The actual values returned for `emit_output` at this
        initializing call are ignored.  Note, this emit structure must be
        consistent across all time steps.

      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
      A tuple `(emit_ta, final_state, final_loop_state)` where:

      `emit_ta`: The RNN output `TensorArray`.
         If `loop_fn` returns a (possibly nested) set of Tensors for
         `emit_output` during initialization, (inputs `time = 0`,
         `cell_output = None`, and `loop_state = None`), then `emit_ta` will
         have the same structure, dtypes, and shapes as `emit_output` instead.
         If `loop_fn` returns `emit_output = None` during this call,
         the structure of `cell.output_size` is used:
         If `cell.output_size` is a (possibly nested) tuple of integers
         or `TensorShape` objects, then `emit_ta` will be a tuple having the
         same structure as `cell.output_size`, containing TensorArrays whose
         elements' shapes correspond to the shape data in `cell.output_size`.

      `final_state`: The final cell state.  If `cell.state_size` is an int, this
        will be shaped `[batch_size, cell.state_size]`.  If it is a
        `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
        If it is a (possibly nested) tuple of ints or `TensorShape`, this will
        be a tuple having the corresponding shapes.

      `final_loop_state`: The final loop state as returned by `loop_fn`.

    Raises:
      TypeError: If `cell` is not an instance of RNNCell, or `loop_fn` is not
        a `callable`.
    """

    if not isinstance(cell, rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not callable(loop_fn):
        raise TypeError("loop_fn must be a callable")

    parallel_iterations = parallel_iterations or 32

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        time = constant_op.constant(0, dtype=dtypes.int32)
        (next_input, elements_finished, initial_state, beam_seq,
         beam_prob, emit_ta, init_loop_state) = loop_fn(None, None, None, None, None, None, None)
            # cell_output, cell_state, beam_seq, beam_prob, finished, emit_ta, loop_state

        flat_input = nest.flatten(next_input)
        # Need a surrogate loop state for the while_loop if none is available.
        loop_state = (init_loop_state if init_loop_state is not None
                      else constant_op.constant(0, dtype=dtypes.int32))

        input_shape = [input_.get_shape() for input_ in flat_input]
        static_batch_size = input_shape[0][0]

        for input_shape_i in input_shape:
            # Static verification that batch sizes all match
            static_batch_size.merge_with(input_shape_i[0])

        batch_size = static_batch_size.value
        if batch_size is None:
            batch_size = array_ops.shape(flat_input[0])[0]

        nest.assert_same_structure(initial_state, cell.state_size)
        state = initial_state
        flat_state = nest.flatten(state)
        flat_state = [ops.convert_to_tensor(s) for s in flat_state]
        state = nest.pack_sequence_as(structure=state,
                                      flat_sequence=flat_state)
        
        def condition(unused_time, elements_finished, *_):
            return math_ops.logical_not(math_ops.reduce_all(elements_finished))

        def body(time, elements_finished, current_input,
                  state, beam_seq, beam_prob, emit_ta, loop_state):
            """Internal while loop body for raw_rnn.

            Args:
              time: time scalar.
              elements_finished: batch-size vector.
              current_input: possibly nested tuple of input tensors.
              emit_ta: possibly nested tuple of output TensorArrays.
              state: possibly nested tuple of state tensors.
              loop_state: possibly nested tuple of loop state tensors.

            Returns:
              Tuple having the same size as Args but with updated values.
            """
            dummy = array_ops.zeros(shape=[tf.shape(beam_seq)[0], tf.shape(beam_seq)[1], 20], dtype=tf.int32)
            
            (next_output, cell_state) = cell(current_input, state)

            nest.assert_same_structure(state, cell_state)
            nest.assert_same_structure(cell.output_size, next_output)
            #cell_output, cell_state, beam_seq, beam_prob, finished, emit_ta, loop_state
            
            (next_input, elements_finished, next_state, beam_seq,
             beam_prob, emit_ta, next_loop_state) = loop_fn(
                next_output, cell_state, beam_seq, beam_prob, elements_finished, emit_ta, loop_state)
            nest.assert_same_structure(state, next_state)
            nest.assert_same_structure(current_input, next_input)
            # If loop_fn returns None for next_loop_state, just reuse the
            # previous one.
            loop_state = loop_state if next_loop_state is None else next_loop_state

            next_time = time + 1
            return (next_time, elements_finished, next_input,
                    next_state, beam_seq, beam_prob, emit_ta, loop_state)

        returned = control_flow_ops.while_loop(
            condition, body, loop_vars=[
                time, elements_finished, next_input,
                state, beam_seq, beam_prob, emit_ta, loop_state],
                shape_invariants=[time.get_shape(), elements_finished.get_shape(), 
                                  next_input.get_shape(), state.get_shape(), 
                                  tensor_shape.TensorShape(None), beam_prob.get_shape(), 
                                  tensor_shape.TensorShape(None), loop_state.get_shape()],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        emit_ta = returned[6]
        ret_tensor = emit_ta.pack() #(b_sz, decode_limit)
        ret_tensor = ret_tensor[:, 1:]
        
        beam_prob = returned[5]
        beam_seq = returned[4]
        
        return ret_tensor, beam_seq, beam_prob #shape(b_sz, deocde_limit) shape(b_sz, beam_sz, steps) shape(b_sz, beam)
