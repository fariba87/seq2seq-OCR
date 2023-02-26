from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import cv2
from ConFig.Config import ConfigReader
cfg = ConfigReader()
from utils import  create_in_out_decoder
#import tensorflow as tf
#import numpy as np

#charset = np.load('')
#vocab_size = len(charset)+2
#num_timesteps = 25# max_text_length_in_batch
embedding_dim= 1000 # encoder instead of one hot encoding
encoder_dim =128  #GRU num unit
from Models.modules.ResNet import ResNet50
from data_generator import getDataByGenerator
############################################################################################

class Seq2SeqDynamicModel(object):
    def __init__(self, encoder_inputs_tensor,
                 decoder_inputs,
                 decoder_out,
                 target_weights,
                 target_vocab_size,
                 buckets,
                 target_embedding_size,
                 attn_num_layers,
                 attn_num_hidden,
                 forward_only,
                 use_gru):
        self.encoder_inputs_tensor = encoder_inputs_tensor
        self.decoder_inputs = decoder_inputs
        self.target_weights = target_weights
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets

        # Create the internal multi-layer cell for our RNN.
        # single_cell = tf.contrib.rnn.BasicLSTMCell(
    #        attn_num_hidden, forget_bias=0.0, state_is_tuple=False
   #     )
        single_cell = tf.keras.layers.GRUCell(attn_num_hidden)
        # if use_gru:
        #     print("using GRU CELL in decoder")
        #     single_cell = tf.contrib.rnn.GRUCell(attn_num_hidden)
        cell = single_cell

#        if attn_num_layers > 1:
 #           cell = tf.contrib.rnn.MultiRNNCell(
  #              [single_cell] * attn_num_layers, state_is_tuple=False
   #         )

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(lstm_inputs, decoder_inputs, seq_length, do_decode):

            num_hidden = attn_num_layers* attn_num_hidden
            gru = tf.keras.layers.GRU(num_hidden, return_sequences=True, return_state=True)
            lstm_inputs = tf.transpose(lstm_inputs, perm=[1, 0, 2])  # [BS, W, Q]
            ##############################################################################################
            whole_sequence_output, final_state = gru(lstm_inputs)  # [BS , W ,  G] , [BS, G]
            attention_states = tf.transpose(whole_sequence_output, perm=[0, 1, 2])

            #  encoder_inputs = whole_sequence_output#tf.concat([outputs[0], outputs[1]], 2)
            #   attention_states = tf.transpose(encoder_inputs, perm=[1, 0, 2])
            #   attention_states = encoder_inputs# tf.transpose(encoder_inputs, perm=[1, 0, 2])

            initial_state = final_state  # tf.concat(axis=1, values=[states[0], states[1]])
            # lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
            #     num_hidden, forget_bias=0.0, state_is_tuple=False
            # )
            # # Backward direction cell
            # lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
            #     num_hidden, forget_bias=0.0, state_is_tuple=False
            # )
            #
            # (outputs, states) = tf.nn.bidirectional_dynamic_rnn(
            #      lstm_fw_cell, lstm_bw_cell, lstm_inputs,
            #      dtype=tf.float32, time_major=True)
            # encoder_inputs = tf.concat([outputs[0], outputs[1]], 2)
            # attention_states = tf.transpose(encoder_inputs, perm=[1, 0, 2])
            # initial_state = tf.concat(axis=1, values=[states[0], states[1]])
            outputs, _, attention_weights_history = attention_decoder(#embedding_attention_decoder(
                decoder_inputs, initial_state, attention_states, cell,
                #num_symbols=target_vocab_size,
                #embedding_size=target_embedding_size,
                num_heads=1,
                output_size=target_vocab_size,
                #output_projection=None,
              #  feed_previous=do_decode,
                initial_state_attention=False,
                attn_num_hidden=attn_num_hidden)
            return outputs, attention_weights_history

        # Our targets are decoder inputs shifted by one.
        # targets = [decoder_inputs[i + 1]
        #            for i in xrange(len(decoder_inputs) - 1)]
        targets =decoder_out

        softmax_loss_function = None  # default to tf.nn.sparse_softmax_cross_entropy_with_logits

        # Training outputs and losses.
        if forward_only:
            self.output, self.loss, self.attention_weights_history = model_with_buckets_dynamic(
                encoder_inputs_tensor, decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y, z: seq2seq_f(x, y, z, True),
                softmax_loss_function=softmax_loss_function)
        else:
            self.output, self.loss, self.attention_weights_history = model_with_buckets_dynamic(
                encoder_inputs_tensor, decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y, z: seq2seq_f(x, y, z, False),
                softmax_loss_function=softmax_loss_function)

        self.attentions = self.attention_weights_history


# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

    Returns:
        A loop function.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(prev,
                                   output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev
    return loop_function
def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=tf.float32, scope=None,
                      initial_state_attention=False, attn_num_hidden=128):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.

    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
            in order to generate i+1-th input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors of
                shape [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either the i-th element
                of decoder_inputs or loop_function(output {i-1}, i)) as follows.
                First, we run the cell on a combination of the input and previous
                attention masks:
                    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                    output = linear(cell_output, new_attn).
            state: The state of each decoder cell the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: when num_heads is not positive, there are no inputs, or shapes
            of attention_states are not set.
    """
    # # MODIFIED ADD START
    # assert num_heads == 1, 'We only consider the case where num_heads=1!'
    # # MODIFIED ADD END
    # if not decoder_inputs:
    #     raise ValueError("Must provide at least 1 input to attention decoder.")
    # if num_heads < 1:
    #     raise ValueError("With less than 1 heads, use a non-attention decoder.")
    # if not attention_states.get_shape()[1:2].is_fully_defined():
    #     raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
    #                      % attention_states.get_shape())
    # if output_size is None:
    output_size = cell.output_size

    #with tf.variable_scope(scope or "attention_decoder"):
    batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = tf.shape(attention_states)[1]#.get_shape()[1].value
    attn_size = attention_states.get_shape()[2]#.value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    v.append([attention_vec_size])
    kernel_in = np.ones((1,1,attn_size, attention_vec_size))
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    hidden_features.append(tf.nn.conv2d(hidden, kernel , [1,1,1,1], "SAME")) #none ,104,1,128
    # for a in xrange(num_heads):
    #     k = tf.get_variable("AttnW_%d" % a,
    #                         [1, 1, attn_size, attention_vec_size])
    #     hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
    #     v.append(tf.get_variable("AttnV_%d" % a,
    #                              [attention_vec_size]))

    state = initial_state

    # MODIFIED: return both context vector and attention weights
    def attention(query):
        """Put attention masks on hidden using hidden_features and query."""
        # MODIFIED ADD START
        ss = None  # record attention weights
        # MODIFIED ADD END
        ds = []  # Results of attention reads will be stored here.
        for a in xrange(num_heads):
            with tf.variable_scope("Attention_%d" % a):
                #y = linear(query, attention_vec_size, True)
                y = tf.keras.layers.Dense(attention_vec_size)(query)  # none,128
                y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                # Attention mask is a softmax of v^T * tanh(...).
                s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y), [2, 3])
                a = tf.nn.softmax(s)
                ss = a
                # a = tf.Print(a, [a], message="a: ",summarize=30)
                # Now calculate the attention-weighted vector d.
                d = tf.reduce_sum(
                    tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                    [1, 2]
                )
                ds.append(tf.reshape(d, [-1, attn_size]))
        # MODIFIED DELETED return ds
        # MODIFIED ADD START
        return ds, ss
        # MODIFIED ADD END

    outputs = []
    # MODIFIED ADD START
    attention_weights_history = []
    # MODIFIED ADD END
    prev = None
    batch_attn_size = tf.stack([batch_size, attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
        a.set_shape([None, attn_size])
    if initial_state_attention:
        # MODIFIED DELETED attns = attention(initial_state)
        # MODIFIED ADD START
        attns, attn_weights = attention(initial_state)
        attention_weights_history.append(attn_weights)
        # MODIFIED ADD END
    for i, inp in enumerate(decoder_inputs):
        if i > 0:
            tf.get_variable_scope().reuse_variables()
        # If loop_function is set, we use it instead of decoder_inputs.
        if loop_function is not None and prev is not None:
            with tf.variable_scope("loop_function", reuse=True):
                inp = loop_function(prev, i)
        # Merge input and previous attentions into one vector of the right size.
        # input_size = inp.get_shape().with_rank(2)[1]
        # todo: use input_size
        input_size = attn_num_hidden
        #x = linear([inp] + attns, input_size, True)

        # x = linear([inp] + attns, input_size, True)
        x = tf.keras.layers.Dense(input_size)(
            tf.concat([tf.expand_dims(inp, axis=1), attns], axis=1))  # [inp] + attns)  #none,128

        # Run the RNN.
        cell_output, state = cell(x, state)
        # Run the attention mechanism.
        if i == 0 and initial_state_attention:
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True):
                # MODIFIED DELETED attns = attention(state)
                # MODIFIED ADD START
                attns, attn_weights = attention(state)
                # MODIFIED ADD END
        else:
            # MODIFIED DELETED attns = attention(state)
            # MODIFIED ADD START
            attns, attn_weights = attention(state)
            attention_weights_history.append(attn_weights)
            # MODIFIED ADD END

        with tf.variable_scope("AttnOutputProjection"):
            output = linear([cell_output] + attns, output_size, True)
        if loop_function is not None:
            prev = output
        outputs.append(output)

    # MODIFIED DELETED return outputs, state
    # MODIFIED ADD START
    return outputs, state, attention_weights_history
    # MODIFIED ADD END


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, embedding_size, num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=tf.float32, scope=None,
                                initial_state_attention=False,
                                attn_num_hidden=128):
    """RNN decoder with embedding and attention and a pure-decoding option.

    Args:
        decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function.
        num_symbols: Integer, how many symbols come into the embedding.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_size: Size of the output vectors; if None, use output_size.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_symbols] and B has shape
            [num_symbols]; if provided and feed_previous=True, each fed previous
            output will first be multiplied by W and added B.
        feed_previous: Boolean; if True, only the first of decoder_inputs will be
            used (the "GO" symbol), and all other decoder inputs will be generated by:
                next = embedding_lookup(embedding, argmax(previous_output)),
            In effect, this implements a greedy decoder. It can also be used
            during training to emulate http://arxiv.org/abs/1506.03099.
            If False, decoder_inputs are used as given (the standard decoder case).
        update_embedding_for_previous: Boolean; if False and feed_previous=True,
            only the embedding for the first symbol of decoder_inputs (the "GO"
            symbol) will be updated by back propagation. Embeddings for the symbols
            generated from the decoder itself remain unchanged. This parameter has
            no effect if feed_previous=False.
        dtype: The dtype to use for the RNN initial states (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    # with tf.variable_scope(scope or "embedding_attention_decoder"):
    #     with tf.device("/cpu:0"):
    embedding = tf.get_variable("embedding",
                                        [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
    emb_inp = [
            tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
    return attention_decoder(
            emb_inp, initial_state, attention_states, cell, output_size=output_size,
            num_heads=num_heads, loop_function=loop_function,
            initial_state_attention=initial_state_attention, attn_num_hidden=attn_num_hidden)


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        average_across_timesteps: If set, divide the returned cost by the total
            label weight.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
        1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
        ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with tf.name_scope(name, "sequence_loss_by_example",
                       logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # todo(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                target = tf.reshape(target, [-1])
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logit, labels=target)
            else:
                crossent = softmax_loss_function(logits=logit, labels=target)
            log_perp_list.append(crossent * weight)
        log_perps = tf.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = tf.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        average_across_timesteps: If set, divide the returned cost by the total
            label weight.
        average_across_batch: If set, divide the returned cost by the batch size.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
        A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
        ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    with tf.name_scope(name, "sequence_loss", logits + targets + weights):
        cost = tf.reduce_sum(sequence_loss_by_example(
            logits, targets, weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = tf.shape(targets[0])[0]
            return cost / tf.cast(batch_size, tf.float32)

        return cost


def model_with_buckets(encoder_inputs_tensor, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.

    The seq2seq argument is a function that defines a sequence-to-sequence model,
    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

    Args:
        encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
        decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
        targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
        weights: List of 1D batch-sized float-Tensors to weight the targets.
        buckets: A list of pairs of (input size, output size) for each bucket.
        seq2seq: A sequence-to-sequence model function; it takes 2 input that
            agree with encoder_inputs and decoder_inputs, and returns a pair
            consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        per_example_loss: Boolean. If set, the returned loss will be a batch-sized
            tensor of losses for each sequence in the batch. If unset, it will be
            a scalar with the averaged loss from all examples.
        name: Optional name for this operation, defaults to "model_with_buckets".

    Returns:
        A tuple of the form (outputs, losses), where:
            outputs: The outputs for each bucket. Its j'th element consists of a list
                of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
            losses: List of scalar Tensors, representing losses for each bucket, or,
                if per_example_loss is set, a list of 1D batch-sized float Tensors.

    Raises:
        ValueError: If length of encoder_inputsut, targets, or weights is smaller
            than the largest (last) bucket.
    """
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = [encoder_inputs_tensor] + decoder_inputs + targets + weights
    with tf.name_scope(name, "model_with_buckets", all_inputs):
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            bucket = buckets[0]
            encoder_inputs = tf.split(encoder_inputs_tensor, bucket[0], 0)
            encoder_inputs = [tf.squeeze(inp, squeeze_dims=[0]) for inp in encoder_inputs]
            bucket_outputs, attention_weights_history = seq2seq(encoder_inputs[:int(bucket[0])],
                                                                decoder_inputs[:int(bucket[1])],
                                                                int(bucket[0]))
            if per_example_loss:
                loss = sequence_loss_by_example(
                    bucket_outputs, targets[:int(bucket[1])], weights[:int(bucket[1])],
                    average_across_timesteps=True,
                    softmax_loss_function=softmax_loss_function)
            else:
                loss = sequence_loss(
                    bucket_outputs, targets[:int(bucket[1])], weights[:int(bucket[1])],
                    average_across_timesteps=True,
                    softmax_loss_function=softmax_loss_function)

    return bucket_outputs, loss, attention_weights_history

def model_with_buckets_dynamic(encoder_inputs_tensor, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.

    The seq2seq argument is a function that defines a sequence-to-sequence model,
    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

    Args:
        encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
        decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
        targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
        weights: List of 1D batch-sized float-Tensors to weight the targets.
        buckets: A list of pairs of (input size, output size) for each bucket.
        seq2seq: A sequence-to-sequence model function; it takes 2 input that
            agree with encoder_inputs and decoder_inputs, and returns a pair
            consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        per_example_loss: Boolean. If set, the returned loss will be a batch-sized
            tensor of losses for each sequence in the batch. If unset, it will be
            a scalar with the averaged loss from all examples.
        name: Optional name for this operation, defaults to "model_with_buckets".

    Returns:
        A tuple of the form (outputs, losses), where:
            outputs: The outputs for each bucket. Its j'th element consists of a list
                of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
            losses: List of scalar Tensors, representing losses for each bucket, or,
                if per_example_loss is set, a list of 1D batch-sized float Tensors.

    Raises:
        ValueError: If length of encoder_inputsut, targets, or weights is smaller
            than the largest (last) bucket.
    """
    # if len(targets) < buckets[-1][1]:
    #     raise ValueError("Length of targets (%d) must be at least that of last"
    #                      "bucket (%d)." % (len(targets), buckets[-1][1]))
    # if len(weights) < buckets[-1][1]:
    #     raise ValueError("Length of weights (%d) must be at least that of last"
    #                      "bucket (%d)." % (len(weights), buckets[-1][1]))

    # all_inputs = [encoder_inputs_tensor] + decoder_inputs + targets + weights
    # with tf.name_scope(name, "model_with_buckets", all_inputs):
    #    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
    bucket = buckets[0]
    # encoder_inputs = tf.split(encoder_inputs_tensor, bucket[0], 0)
    # encoder_inputs = [tf.squeeze(inp, squeeze_dims=[0]) for inp in encoder_inputs]
    bucket_outputs, attention_weights_history = seq2seq(encoder_inputs_tensor,
                                                        decoder_inputs[:int(bucket[1])],
                                                        bucket[0])
    if per_example_loss:
        loss = sequence_loss_by_example(
            bucket_outputs, targets[:int(bucket[1])], weights[:int(bucket[1])],
            average_across_timesteps=True,
            softmax_loss_function=softmax_loss_function)
    else:
        loss = sequence_loss(
            bucket_outputs, targets[:int(bucket[1])], weights[:int(bucket[1])],
            average_across_timesteps=True,
            softmax_loss_function=softmax_loss_function)

    return bucket_outputs, loss, attention_weights_history

##################################################################################
def get_data_and_model(dataset = 'MJsyn', mode ="Transformer"):

    data_gen, Maxlen, lenvoc , vocab = getDataByGenerator(dataset= dataset ,mode =mode )#'MJsyn')
    w_max=800
  #  fullmodel = FULL_FE_TRANS(w_max, lenvocab=lenvoc + 2, maxlen=Maxlen + 2)
  #  model_fe_tr = fullmodel().model
    return data_gen , Maxlen,lenvoc, vocab #, model_fe_tr ,fullmodel ,

data_gen ,Maxlen , lenvoc , vocab = get_data_and_model()

class Model_FE_Att(tf.keras.Model):
    def __init__(self, lenvoc):
        super(Model_FE_Att, self).__init__()
        self.lenvoc =lenvoc
        self.target_embedding_size = 10
        self.attn_num_layers = 2
        self.attn_num_hidden = 128
        self.Feat = ResNet50()
        self.is_training =True
        self.use_gru= True



        #self.encoder_in = tf.keras.layers.Input()
        self.classifier = tf.keras.layers.Dense(self.lenvoc + 1, activation='softmax')
        self.encoder_in= tf.keras.layers.Input((64, None, 1),
                                                dtype='float32')  # (None, 64, 3))  # base on Tensorflow backend
        self.decoder_in = tf.keras.layers.Input(name='the_labels', shape=[None],
                                            dtype='float32')  # , ragged=True )# [None] label_shape
        self.decoder_out = tf.keras.layers.Input(name='the_labels', shape=[None],)
        #                                     dtype='float32')  # , ragged=True )# [None] label_shape
        self.target_weights = tf.keras.layers.Input(name='the_labels', shape=[None],
                                            dtype='float32')
        self.optimizer = tf.keras.optimizers.Adam(0.00001)
        # self.encoder_size = int(math.ceil(1. * self.config.targetWidth / 4))

        #self.input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')  # , ragged=True )
        #self.label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')  # , ragged=True )

    def __call__(self):
        cnn_out = self.Feat(self.encoder_in)
        self.encoder_size = tf.cast(tf.math.ceil(tf.divide(tf.cast(tf.shape(cnn_out)[1], tf.float32), tf.cast(cfg.SeqDivider, tf.float32))), tf.int32)# tf.constant(cfg.SeqDivider, dtype=tf.int32))),
                #dtype=tf.int32)
        self.decoder_size = self.lenvoc+2#config.max_prediction_length + 2
        self.buckets = [(self.encoder_size, self.decoder_size)]

        self.perm_conv_output = tf.transpose(cnn_out, perm=[1, 0, 2])
        self.attention_decoder_model = Seq2SeqDynamicModel(
            encoder_inputs_tensor=self.perm_conv_output,
            decoder_inputs=self.decoder_in,
            decoder_out=self.decoder_out,
            target_weights=self.target_weights,
            target_vocab_size=self.lenvoc+3,#config.num_classes,
            buckets=self.buckets,
            target_embedding_size=self.target_embedding_size,
            attn_num_layers=self.attn_num_layers,
            attn_num_hidden=self.attn_num_hidden,
            forward_only=not (self.is_training),
            use_gru=self.use_gru)
        output = self.attention_decoder_model.output
        model1 = tf.keras.Model(inputs=[self.encoder_in, self.decoder_in, self.decoder_out, self.target_weights],
                               outputs=[output])
        self.model =model1
        self.loss = self.attention_decoder_model.loss
        return self
modelAtt = Model_FE_Att(lenvoc=lenvoc)
model = modelAtt().model
loss = modelAtt.loss
model([])
from Models.modules.callbacks import earlystopping, lr_scheduler_tr, tensorboard_cb , CustomSchedule
##########################################################################################################
def apply_gradient(model , x, y):
    with tf.GradientTape() as tape:
        logits = model(x)#,yin)
        GO_ID = 1
        EOS_ID = 2

        table = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value="",
            checkpoint=True,
        )

        insert = table.insert(
            tf.constant(list(range(lenvoc)), dtype=tf.int64), #self.config.num_classes
            tf.constant(['','','']+vocab),
        )


        trans_ground = tf.cast(tf.transpose(y), tf.int64) #####################decoder_inputs
        trans_ground = tf.map_fn(
            lambda m: tf.foldr(
                lambda a, x: tf.cond(
                    tf.equal(x, EOS_ID),
                    lambda: '',
                    lambda: table.lookup(x) + a  # pylint: disable=undefined-variable
                ),
                m,
                initializer=''
            ),
            trans_ground,
            dtype=tf.string
        )
        ground = tf.cond(
            tf.equal(tf.shape(trans_ground)[0], 1),
            lambda: trans_ground[0],
            lambda: trans_ground,
        )

        num_feed = []
        prb_feed = []


        for line in xrange(len(logits)):#self.attention_decoder_model.output)):
            guess = tf.argmax(logits[line], axis=1)
            proba = tf.reduce_max(
                tf.nn.softmax(logits[line]), axis=1)
            num_feed.append(guess)
            prb_feed.append(proba)

        # Join the predictions into a single output string.
        trans_output = tf.transpose(num_feed)
        trans_output = tf.map_fn(
            lambda m: tf.foldr(
                lambda a, x: tf.cond(
                    tf.equal(x, EOS_ID),#lenvoc+2),#self.config.EOS_ID),
                    lambda: '',
                    lambda: table.lookup(x) + a  # pylint: disable=undefined-variable
                ),
                m,
                initializer=''
            ),
            trans_output,
            dtype=tf.string
        )

        # Calculate the total probability of the output string.
        trans_outprb = tf.transpose(prb_feed)
        trans_outprb = tf.gather(trans_outprb, tf.range(tf.size(trans_output)))
        trans_outprb = tf.map_fn(
            lambda m: tf.foldr(
                lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
                m,
                initializer=tf.cast(1, tf.float64)
            ),
            trans_outprb,
            dtype=tf.float64
        )

        prediction = tf.cond(
            tf.equal(tf.shape(trans_output)[0], 1),
            lambda: trans_output[0],
            lambda: trans_output,
        )
        probability = tf.cond(
            tf.equal(tf.shape(trans_outprb)[0], 1),
            lambda: trans_outprb[0],
            lambda: trans_outprb,
        )

        prediction = tf.identity(prediction, name='prediction')
        probability = tf.identity(probability, name='probability')


        loss_val = loss(real =y, pred =logits)# tar_real
        #accuracy = accuracy_function_transformer(real=y , pred=logits)#tar_real
        accuracy = 1 - tf.reduce_mean(tf.edit_distance(tf.string_split(prediction), tf.string_split(ground)))
        # print('accuracy', accuracy)
        #variables = model.trans.trainable_variables + model.FEnew.trainable_variables  #model.trans :transformer - model1: embedding
        variables = model.trainable_variables
        gradients = tape.gradient(loss_val ,variables)
      #  gradients = tf.clip_by_value(gradients, -1., 1.)
        modelAtt.optimizer.apply_gradients(zip(gradients, variables)) #fullmodel
    return logits, loss_val, accuracy
##########################################################################################################

numstep = np.int32(np.divide(data_gen.__len__(), cfg.batchSize))
def train_data_for_one_epoch():
    losses =[]
    accBatch =[]
    for step in range(numstep):#56):#, (x_batch_train , y_batch_train) in enumerate(trainDataset):
        Xin, yAtt, yCTC, encoder_mask, times, w_max = data_gen.__getitem__(step)
        yin, tar_real = create_in_out_decoder(yAtt, lenvoc)
        x_batch_train = (Xin, yin, encoder_mask)#X_batch_resized , y_batch_resized_Attn, encoder_mask)
        y_batch_train = tar_real#y_batch_resized_Attn
        x_batch_train = (Xin, yin,tar_real, encoder_mask)#X_batch_resized , y_batch_resized_Attn, encoder_mask)

        logits, loss_val, acc = apply_gradient(model, x_batch_train, y_batch_train)
        losses.append(loss_val)
        accBatch.append(acc)
    return losses, accBatch
##########################################################################################################
_callbacks = [earlystopping, lr_scheduler_tr ]#, tensorboard_cb]
callbacks = tf.keras.callbacks.CallbackList(_callbacks, add_history=True, model=model)
logs = {}
train_loss_history=[]
train_acc_history =[]
def train_transformer(epochs):
    for epoch in range(epochs):#cfg.TotalEpoch):
        #callback
        callbacks.on_epoch_begin(epoch, logs=logs)

        losses_train, acc_train = train_data_for_one_epoch()

        acc_train_mean = np.mean(acc_train)
        train_acc_history.append(acc_train_mean)

        losses_train_mean = np.mean(losses_train)
        train_loss_history.append(losses_train_mean)

        print('epoch {} :loss {} and accuracy {}'.format(epoch + 1, losses_train_mean, acc_train_mean))  # , acc_train_mean)
        #print('accuracy in epoch {} is {}'.format(epoch + 1, acc_train_mean))
    return train_loss_history, train_acc_history# losses_train_mean
train_loss_history, train_acc_history = train_transformer(cfg.TotalEpoch)



def attention_with_keras(checkpoint_dir, encoder_in, decoder_in, decoder_out, vocab_size, maxlen, batch_size):

    class Encoder(tf.keras.Model):
        def __init__(self,# vocab_size,
                     #num_timesteps,
                     #embedding_dim,
                     encoder_dim, **kwargs):
            super(Encoder, self).__init__(**kwargs)
           # self.encoder_dim = encoder_dim
           # self.embedding = tf.keras.layers.Embedding(
           # vocab_size, embedding_dim, input_length=num_timesteps)
            self.rnn = tf.keras.layers.GRU(  #with attention return =True
            encoder_dim, return_sequences=True, return_state=True)

        def call(self, x, state):
           # x = self.embedding(x)

            x, state = self.rnn(x, initial_state=state)
            return x, state

        def init_state(self, batch_size):
            return tf.zeros((batch_size, self.encoder_dim))  #zeros initial state

    class Attention(tf.keras.layers.Layer):
        def __init__(self, num_units):  #
            super(Attention, self).__init__()
            self.W1 = tf.keras.layers.Dense(num_units)
            self.W2 = tf.keras.layers.Dense(num_units)
            self.V = tf.keras.layers.Dense(1)  # i dont know why 1
        def call(self, query, values):  #
            # query: hidden state of decoder

            # query.shape: (batch_size, num_units)
            # values are encoder states at every timestep i
            # values.shape: (batch_size, num_timesteps, num_units)
            # add time axis to query: (batch_size, 1, num_units)
            query_with_time_axis = tf.expand_dims(query, axis=1)
            # compute score:
            score = self.V(tf.keras.activations.tanh(
                self.W1(values) + self.W2(query_with_time_axis)))
            # compute softmax
            alignment = tf.nn.softmax(score, axis=1)  # it is weight
            # compute attended output
            context = tf.reduce_sum(  # weighted sum
                tf.linalg.matmul(
                    tf.linalg.matrix_transpose(alignment),
                    values
                ), axis=1
            )
            context = tf.expand_dims(context, axis=1)
            return context, alignment
    # query is the decoder state at time step j
    class Decoder(tf.keras.Model):
        def __init__(self, vocab_size,
                     embedding_dim,
                     num_timesteps,
                     decoder_dim,
                     **kwargs):
            super(Decoder, self).__init__(**kwargs)
            self.decoder_dim = decoder_dim  # GRU unit
            self.attention = Attention(embedding_dim)
            # self.attention = Attention(embedding_dim)
            #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=num_timesteps)
            self.rnn = tf.keras.layers.GRU(decoder_dim, return_sequences=True, return_state=True)
            self.Wc = tf.keras.layers.Dense(decoder_dim, activation="tanh")
            self.Ws = tf.keras.layers.Dense(vocab_size)

        def call(self, x, state, encoder_out):
            #x = self.embedding(x) # first embedding if it is needed
            context, alignment = self.attention(x, encoder_out) # then attention
            x = tf.expand_dims(tf.concat([x, tf.squeeze(context, axis=1)], axis=1), axis=1)   # x is contatenation of x and context
            x, state = self.rnn(x, state)  #
            x = self.Wc(x)
            x = self.Ws(x)   # with the size of vocabulary (number of classes)
            return x, state, alignment
    embedding_dim = 256
    encoder_dim, decoder_dim = 1024, 1024
    #vocab_size_en # is needed for embedding(machine translation) : i can ignore it
    encoder = Encoder(embedding_dim)#vocab_size_en+1, embedding_dim, maxlen_en, encoder_dim)  # maxlen_en is also can be ignored
   # vocab_size_fr+1= 100
    decoder = Decoder(vocab_size=vocab_size+1, embedding_dim=embedding_dim, num_timesteps=maxlen, decoder_dim=decoder_dim)
    def loss_fn(ytrue, ypred):
        scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(ytrue, 0))  # create a mask to avoid zeros padding in computing loss
        mask = tf.cast(mask, dtype=tf.int64)
        loss = scce(ytrue, ypred, sample_weight=mask)
        return loss
    batch_size =10
    # encoder_in : output from CNN (feature maps)[None, W, C]
    # decoder in : SOS+ text label  (padded)
    # decoder out : text label +EOS  (padded) [Bs, W, num_class]

    # for encoder_in, decoder_in, decoder_out in train_dataset:
    #     encoder_state = encoder.init_state(batch_size)
    #     encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    #     decoder_state = encoder_state
    #     decoder_pred, decoder_state = decoder(decoder_in, decoder_state)
    #     break
   # for encoder_in, decoder_in, decoder_out in train_dataset:
    encoder_state = encoder.init_state(batch_size)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state
    decoder_pred, decoder_state = decoder(decoder_in, decoder_state)

        #break
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state
    loss = 0

    for t in range(decoder_out.shape[1]):
        decoder_in_t = decoder_in[:, t]
        decoder_pred_t, decoder_state, _ = decoder(decoder_in_t, decoder_state, encoder_out)
        loss += loss_fn(decoder_out[:, t], decoder_pred_t)
    Feat = ResNet50()
    @tf.function
    def train_step(encoder_in, decoder_in, decoder_out, encoder_state):
        with tf.GradientTape() as tape:
            Fmaps=Feat(encoder_in)
            encoder_in1 = tf.transpose(Fmaps ,(1,0,2))
            encoder_out, encoder_state = encoder(encoder_in1, encoder_state)
            decoder_state = encoder_state
            loss = 0
            for t in range(decoder_out.shape[1]):
                decoder_in_t = decoder_in[:, t]
                decoder_pred_t, decoder_state, _ = decoder(decoder_in_t,
                decoder_state, encoder_out)
                loss += loss_fn(decoder_out[:, t], decoder_pred_t)
        variables = (encoder.trainable_variables + decoder.trainable_variables)
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss / decoder_out.shape[1]

    optimizer = tf.keras.optimizers.Adam()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,  encoder=encoder, decoder=decoder)
    num_epochs = 250
    eval_scores = []
    for e in range(num_epochs):
        encoder_state = encoder.init_state(batch_size)
        #for batch, data in enumerate(train_dataset):
        for step in range(1):
            Xin, yAtt, yCTC, encoder_mask, times, w_max = data_gen.__getitem__(step)
            yin, tar_real = create_in_out_decoder(yAtt, lenvoc)
            x_batch_train = (Xin, yin, encoder_mask)  # X_batch_resized , y_batch_resized_Attn, encoder_mask)
            encoder_in = Xin
            decoder_in = yin
            decoder_out = y_batch_train = tar_real  # y_batch_resized_Attn
            
           # encoder_in, decoder_in, decoder_out = data
            # print(encoder_in.shape, decoder_in.shape, decoder_out.shape)
            loss = train_step( encoder_in, decoder_in, decoder_out, encoder_state)
        print("Epoch: {}, Loss: {:.4f}".format(e + 1, loss.numpy()))
        if e % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        #predict(encoder, decoder, batch_size, sents_en, data_en, sents_fr_out, word2idx_fr, idx2word_fr)
       # eval_score = evaluate_bleu_score(encoder, decoder,test_dataset, word2idx_fr, idx2word_fr)
        #print("Eval Score (BLEU): {:.3e}".format(eval_score))

    # eval_scores.append(eval_score)
    checkpoint.save(file_prefix=checkpoint_prefix)
#attention_with_keras(checkpoint_dir, vocab_size=lenvoc, maxlen=Maxlen, batch_size=8)