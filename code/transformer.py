import math
import numpy as np
import tensorflow as tf


class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        STUDENT MUST WRITE:

        Computes attention given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys

        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        # 1) compute attention weights using queries and key matrices 
        # 2) return the attention matrix
        K_t = tf.transpose(K, perm=[0, 2, 1])

        attn_mat = Q @ K_t
        embedding_size = float(K.shape[-1])
        attn_mat /= tf.sqrt(embedding_size)

        if self.use_mask:
            attn_mat += atten_mask
        attn_mat = tf.nn.softmax(attn_mat, axis=-1)

        return attn_mat



class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # Initialize the weight matrices for K, V, and Q.

        self.K = self.add_weight(shape=(input_size, output_size), initializer='random_normal', name = "K")
        self.Q = self.add_weight(shape=(input_size, output_size), initializer='random_normal', name = "Q")
        self.V = self.add_weight(shape=(input_size, output_size), initializer='random_normal', name = "V")
        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # - Apply 3 matrix products to turn inputs into keys, values, and queries. 
        # - Call your AttentionMatrix layer with the keys and queries.
        # - Apply the attention matrix to the values.
    
        keys = tf.tensordot(inputs_for_keys, self.K, axes=[[2], [0]])
        vals = tf.tensordot(inputs_for_values, self.V, axes=[[2], [0]])
        queries = tf.tensordot(inputs_for_queries, self.Q, axes=[[2], [0]])

        attn_mat = self.attn_mtx((keys, queries))
        output = attn_mat @ vals
        return output


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        head_sz = emb_sz // 3
        self.head_1         = AttentionHead(emb_sz, head_sz, is_self_attention=use_mask)
        self.head_2         = AttentionHead(emb_sz, head_sz, is_self_attention=use_mask)
        self.head_3         = AttentionHead(emb_sz, head_sz, is_self_attention=use_mask)
        
        self.W_0 = self.add_weight(shape=(head_sz * 3, emb_sz), initializer='random_normal', name = "W_0", trainable=True)
    
    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        
        out_1 = self.head_1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        out_2 = self.head_2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        out_3 = self.head_3(inputs_for_keys, inputs_for_values, inputs_for_queries)
        out = tf.concat([out_1, out_2, out_3], axis=2, name='concat')
        out = tf.tensordot(out, self.W_0, axes=[[2], [0]])
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        self.ff_layer = tf.keras.layers.Dense(emb_sz, activation = 'relu')

        self.self_atten         = AttentionHead(emb_sz, emb_sz, is_self_attention=True)  if not multiheaded else MultiHeadedAttention(emb_sz, True)
        self.self_context_atten = AttentionHead(emb_sz, emb_sz, is_self_attention=False) if not multiheaded else MultiHeadedAttention(emb_sz, False)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)

    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        1) compute MASKED attention on the inputs
        2) residual connection and layer normalization
        3) computed UNMASKED attention using context
        4) residual connection and layer normalization
        5) feed forward layer
        6) residual layer and layer normalization
        7) return relu of tensor
        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """

        agg_1 = self.self_atten(inputs, inputs, inputs)
        agg_1 = self.layer_norm(agg_1 + inputs)
        agg_2 = self.self_context_atten(context_sequence, context_sequence, agg_1)
        agg_2 = self.layer_norm(agg_1 + agg_2)

        agg_3 = self.ff_layer(agg_2)
        agg_3 = self.layer_norm(agg_3 + agg_2)
        return tf.nn.relu(agg_3)


def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        out = self.embedding(x)

        out *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        return out + self.pos_encoding
    