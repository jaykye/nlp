import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf


# transformer architecture 만드는 부분만 떼어왔다.

# Positional encoding
# Embedding matrix에 진짜 말 그대로 더할꺼야.
# Embedding matrix shape: (vocab size, embed_dim) -> 근데 현재 문장에 맞게 (문장길이 * dim)

def get_angles(pos, i, d_model):
    '''

    :param pos: array, n * 1
    :param i: array, 1 * n
    :param d_model: int
    :return: array,  seq_len(pos) * embed_dim(d_model)
    '''
    # where pos is position in sentence and i is the index of dimension.
    # 문장 속 각 단어의 embedding dim 마다 값을 하나씩 부여함.
    # embedding은 각 단어의 여러 속뜻을 대표함. 각 속뜻이 문장 내에서 어떤 위치에 있냐에 따라서 의미 변화를 수치화 하는 작업.

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    '''
    encode the position with sin and cos.
    :param position: int, the size of features. (단어 갯수)
    :param d_model: int, embedding dimension.
    :return:
    '''
    # 단어갯수 * embed_dim == embedding layer shape
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]  # np.array([angles]) 와 같다. 그냥 axis 하나 날아주는 것.
    # this is 3D, with length of 1 that has 2D matrix. 아,... 나중에 batch로 한꺼번에 여러문장 쓰려고 3d로 만드는구나...
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq_batch):
    # Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input.
    # The mask indicates where pad value `0` is present: it outputs a `1` at those locations, and a `0` otherwise.
    # seq = tf.cast(seq == 0, tf.float32)

    # padding이 0을 사용해서 만드는 것 같네.
    seq_batch = tf.cast(tf.math.equal(seq_batch, 0), tf.float32)  # 0인 부분 찾아서 True/False return 후, float으로 cast.
    # add extra dimensions to add the padding to the attention logits.
    return seq_batch[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    # 미래정보를 가려주는 마스크를 생성
    # The look-ahead mask is used to mask the future tokens in a sequence.
    # In other words, the mask indicates which entries should not be used.
    # This means that to predict the third word, only the first and second word will be used.
    # Similarly to predict the fourth word, only the first, second and the third word will be used and so on.
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # band part는 banded matrix라는 매트릭스를 만들어 주는데, 그냥 (i,i) 대각선 주위만 남기고 나머지는 0으로 만들어주는 method임.
    # 위의 경우, row부분은 그냥 내버려두고, column 0번 오른쪽만 0으로 만들라는 소리.
    # 문장에서 i번째 단어의 경우, 이 마스크 매트릭스에서 i번째 row를 꺼내서 쓰면 되겠다.
    return mask  # (seq_len, seq_len)


  ## Scaled dot product attention

    # The attention function used by the transformer takes three inputs: Q (query), K (key), V (value).
    # The dot-product attention is scaled by a factor of square root of the depth (dimension of keys).
    # This is done because for large values of depth, the dot product grows large in magnitude,
    # pushing the softmax function where it has small gradients resulting in a very hard softmax.
    # For example, consider that `Q` and `K` have a mean of 0 and variance of 1.
    # Their matrix multiplication will have a mean of 0 and variance of `dk`.
    # So the *square root of `dk`* is used for scaling, so you get a consistent variance regardless of
    # the value of `dk`. If the variance is too low the output may be too flat to optimize effectively.
    # If the variance is too high the softmax may saturate at initilization making it dificult to learn.
    # The mask is multiplied with -1e9 (close to negative infinity).
    # This is done because the mask is summed with the scaled matrix multiplication of Q and K and is applied
    # immediately before a softmax. The goal is to zero out these cells, and large negative inputs to softmax
    # are near zero in the output.

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    # (각 word 에서의 q * question 갯수) * k
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k), 이거 같은 값일껄?
    # (각 word 에서의 모든 question에 대한 k의 합)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # 마스크 값은 원래 -inf 로 해서 softmax 적용할 때 0 나오게 한다.

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# As the softmax normalization is done on K, its values decide the amount of importance given to Q.
# The output represents the multiplication of the attention weights and the V (value) vector.
# This ensures that the words you want to focus on are kept as-is and the irrelevant words are flushed out.

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

# ## Multi-head attention
# 왜 여러개의 attention이 필요한가?
# 예문: 내가 오늘 레스토랑에서 친구와 밥을 맛있게 먹었다.
# 여기서 레스토랑을 보자. 얘는 다른 단어들을 보면서 attention을 계산한다.
# 레스토랑은 명사니깐 동사가 중요하다! 이런 한가지 규칙으로 찾는다.
# 그러면 사실 좋은 결과는 안나옴. 그래서 각 다른 생각을가진 attention head 가 필요하다. (장소네? -> 무엇을 했나?, 누가 갔나)
# 간단히 밀해서 각 단에마다 attention head 하나씩 할당해서 dim 만큼의 attention 벡터를 계산한 후, weighted average 계산.
# Multi-head attention consists of four parts:
# *    Linear layers and split into heads.
# *    Scaled dot-product attention.
# *    Concatenation of heads.
# *    Final linear layer.

# Each multi-head attention block gets three inputs; Q (query), K (key), V (value).
# These are put through linear (Dense) layers and split up into multiple heads.
# The `scaled_dot_product_attention` defined above is applied to each head (broadcasted for efficiency).
# An appropriate mask must be used in the attention step.
# The attention output for each head is then concatenated (using `tf.transpose`, and `tf.reshape`)
# and put through a final `Dense` layer.
#
# Instead of one single attention head, Q, K, and V are split into multiple heads
# because it allows the model to jointly attend to
# information at different positions from different representational spaces.
# After the split, each head has a reduced dimensionality, so the total computation cost
# is the same as a single head attention with full dimensionality.

class MultiHeadAttention(tf.keras.layers.Layer):  # 이거는 layer 네?
    def __init__(self, d_model, num_heads):
        # 사실 sequence embedding 이 들어오면 embedding dimension 보면 d_model 은 따로 넣어줄 필요가 없는데,
        # 여기서는 처음 q, k, v linear activation 때문에 필요하대...
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # h
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # depth 가 하나의 head 에 할당할 dimension size.

        # 왜 여기서 d_model 개나 필요해?  -> d_model 유지하려면 (shape 유지하려면 필수)
        # 보통 qkv 는 (seq_len, embedding dim) 으로 들어오는데...
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        말 그대로 마지막 dimension, 원래 d_model 이었던 column 갯수를 그냥 (num_heads, depth)로 나눠주는 것.
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # (batch_size, seq_len_q, num_heads, depth)
        # 각 head 에서 num_heads * depth 가 필요하므로 transpose가 필요하다.
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len_q, depth)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # functional model 시작.
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 다시 원래대로 transpose 해주네?
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        # multi-head 때문에 분산됬던 것 다시 원래대로 모아준다.
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

# ## Point wise feed forward network

# Point wise feed forward network consists of two fully-connected layers with a ReLU activation in between.


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)  # 다시 shape 원위치
    ])


# ## Encoder and decoder

# The transformer model follows the same general pattern as a standard [sequence to sequence with attention model].
# input 문장은, 인코더 레이어 N개를 지나면서 각 단어 마다 output 하나를 만든다.
# The decoder attends on the encoder's output and its own input (self-attention) to predict the next word.

# ### Encoder layer
#
# Each encoder layer consists of sublayers:
#
# 1.   Multi-head attention (with padding mask)
# 2.    Point wise feed forward networks. (두개의 dense layer 인데, 첫번째는 arbitrary dff 개의 unit, 두번째는 input 의
#       shape 를 유지하기 위해서, 항상 d_model 개의 unit이 필요하다.(matters for output dimension.)
#
# 각 sublayer 끝에는 하나의 residual connection과, layer normalization이 붙어있다. - Residual connection은
# vanishing gradient 문제를 피하도록 도와준다.
#
# The output of each sublayer is `LayerNorm(x + Sublayer(x))`.
# The normalization is done on the `d_model` (last) axis. There are N encoder layers in the transformer.

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # dropout은 모든 sublayer 끝에 달려있으며, normalization  전에 한다.
        # 또한, dropuout은 embedding과 positional encoding 의 sum에도 적용된다. 사용되는 P_drop 값은 0.1 이다.
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        
        # 밑에 x는 multihead를 skip한 x임. Weight없이 그냥 더하는구나.
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
# ### Decoder layer
#
# Each decoder layer consists of three sublayers:
#
# 1.   Masked multi-head attention (with look ahead mask and padding mask)
# 2.   Multi-head attention (with padding mask). V (value) and K (key) receive the ENCODER output as inputs.
#      Q (query) receives the output from the masked multi-head attention sublayer.
# 3.   Point wise feed forward networks
#
# Each of these sublayers has a residual connection around it followed by a layer normalization.
# The output of each sublayer is `LayerNorm(x + Sublayer(x))`.
# The normalization is done on the `d_model` (last) axis.
#
# There are N decoder layers in the transformer.
#
# As Q receives the output from decoder's first attention block, and K receives the encoder output,
# the attention weights represent the importance given to the decoder's input based on the encoder's output.
# In other words, the decoder predicts the next word by looking at the encoder output and self-attending
# to its own output.
# See the demonstration above in the scaled dot product attention section.

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # epsilon: 분모 sqrt에서 variance에 더해서 numerical stability.
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # 첫 attention block 은 self attention block인데, training 단계에서는 future 데이터가 포함되어있다. -- 가려줘야함.
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        # 밑에 x는 multihead 를 skip 한 x임. Weight 없이 그냥 더하는구나.
        out1 = self.layernorm1(attn1 + x)

        # 두번째 layer 에서는 Encoder 데이터 (원본언어 데이터)를 사용한다. 얘네는 길이가 고정되어 있어서 padding이 적용되어 있다.
        # 따라서 padding mask 를 사용해 줘야한다.
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # residual connection + normalization
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

# ### Encoder
#
# The `Encoder` consists of:
# 1.   Input Embedding
# 2.   Positional Encoding
# 3.   N encoder layers
#
# The input is put through an embedding which is summed with the positional encoding.
# The output of this summation is the input to the encoder layers.
# The output of the encoder is the input to the decoder.


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)  # 각 vocab에 10개의 dimension을 붙여줌.
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        # N개의 encoder는 functional 하게 만들거임.
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        # 애초에 embedding 된 애를 쓰는게 아닌가?? 이게 필요한가보네?? x가 그냥 token sequence 형태로 들어오나봐?
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # paper는 embedding에다가 sqrt(d_model) 곱한다.
        x += self.pos_encoding[:, :seq_len, :]
        # 이거 3D 이므로 모든 batch에 broadcasting 가능.
        # 두번째 dimension은 maximum_position_encoding 으로 여분의 encoding을 만든 상태라 실제 seq_len 까지만 잘라서 써준다.

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

# ### Decoder

#  The `Decoder` consists of:
# 1.   Output Embedding
# 2.   Positional Encoding
# 3.   N decoder layers
#
# The target is put through an embedding which is summed with the positional encoding.
# The output of this summation is the input to the decoder layers.
# The output of the decoder is the input to the final linear layer.


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        # 아 target 문장이 얼마나 길어질 지 몰라서 이렇게 하는 건가??

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]  # 0 - batch size
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

# ## Create the Transformer

# Transformer consists of the encoder, decoder and a final linear layer.
# The output of the decoder is the input to the linear layer and its output is returned.

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        # pe_input, pe_target 은 max positional encoding len
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
   pass

else:

    td = np.array([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])

    # tf.constant == np.array나 마찬가지.
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    create_padding_mask(x)
    tf.linalg.band_part(np.ones((4,4)), 0, -1)

    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    temp
    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`, so the second `value` is returned.
    # 그냥 우연히 두번째 row만 빼고 다 0곱해져서 없어짐.
    # 이게 얼마나 현재 q에 relevant한 k인지 알아보는 과정 + value에다가도 같은 작업해서
    # 결국에는 현재 q에 관련된 k와 제일 관련된 v값을 찾아서 softmax 로 logit으로 만들어주고 결과 matrix를
    # Attention으로 사용하는 것. 결과적으로 v sequence에서 각 q row에(dim) 제일 중요한 단어를 찾아 주는 셈.
    temp_q = tf.constant([[0, 12, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    # This query aligns with a repeated key (third and fourth),
    # so all associated values get averaged.
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    # This query aligns equally with the first and second key,
    # so their values get averaged.
    temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)
    # Pass all the queries together.

    temp_q = tf.constant([[0, 0, 10],
                          [0, 10, 0],
                          [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    print_out(temp_q, temp_k, temp_v)

    # Create a `MultiHeadAttention` layer to try out.
    # At each location in the sequence, `y`, the `MultiHeadAttention` runs all 8 attention heads across
    # all other locations in the sequence, returning a new vector of the same length at each location.

    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(v=y, k=y, q=y, mask=None)
    out.shape, attn.shape

    sample_ffn = point_wise_feed_forward_network(512, 2048)
    sample_ffn(tf.random.uniform((64, 50, 512))).shape

    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)


    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)

    sample_decoder_layer_output.shape  # (batch_size, target_seq_len, d_model)




    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab_size=8500,
                             maximum_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000,
                             maximum_position_encoding=5000)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    output.shape, attn['decoder_layer2_block2'].shape




    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)

    # ## Set hyperparameters

    # To keep this example small and relatively fast, the values for *num_layers, d_model, and dff* have been reduced.
    # *num_layers=6*, *d_model = 512*, *dff = 2048*.
    # Note: By changing the values below, you can get the model that achieved state of the art on many tasks.

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    # ## Optimizer

    # Use the Adam optimizer with a custom learning rate scheduler.

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    temp_learning_rate_schedule = CustomSchedule(d_model)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    # ## Loss and metrics

    # Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
    # 아, target sequences도 padding이 되어 있어서 loss에다가도 mask를 씌어줘야 되는구나.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    # loss_object([1, 0, 1], [[1.2, 0, 2], [1.2, 0, 2], [1.2, 0, 2]])
    # label, and logits

    def loss_function(real, pred):
        # real values needs to be padded firsrt.
        mask = tf.math.logical_not(tf.math.equal(real, 0))  # 그냥 ~(real == 0)와 같은 것.
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)  # probably is just tf.float32
        loss_ *= mask  # loss 계산 이후에 mask 적용.
        # tf.reduce_sum은 그냥 매트릭스 안에 있는 값들의 총합을 계산한다.
        # loss는 실제 probability logit과 classification 값의 차이를 계산.
        # loss의 총합을 가지고 실제 prediction (mask로 없어진 애들 제외하고) 당 loss를 계산.
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


    train_loss = tf.keras.metrics.Mean(name='train_loss')  # computes the weighted mean of given values.
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # ## Training and checkpointing

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size(),
        target_vocab_size=tokenizers.en.get_vocab_size(),
        pe_input=1000,
        pe_target=1000,
        rate=dropout_rate)


    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)  # input을 보고 값이 0 인부분을 찾아내서 (batch, 1, 1, seq_len) 짜리 마스크냄.

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask


    # Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every `n` epochs.

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # The target is divided into tar_inp and tar_real.
    # tar_inp is passed as an input to the decoder. `tar_real` is that same input shifted by 1:
    # At each location in `tar_input`, `tar_real` contains the  next token that should be predicted.
    # For example, `sentence` = "SOS A lion in the jungle is sleeping EOS"
    # `tar_inp` =  "SOS A lion in the jungle is sleeping"
    # `tar_real` = "A lion in the jungle is sleeping EOS"
    #
    # The transformer is an auto-regressive model: it makes predictions one part at a time,
    # and uses its output so far to decide what to do next.
    # During training this example uses teacher-forcing.
    # Teacher forcing is passing the true output to the next time step regardless of what the model predicts
    # at the current time step.
    # As the transformer predicts each word, *self-attention* allows it to look at the previous words
    # in the input sequence to better predict the next word.
    # To prevent the model from peeking at the expected output the model uses a look-ahead mask.

    EPOCHS = 20

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    # tf.function function 하나를 TensorFlow Graph 로 바꿔줌.
    # An "input signature" can be optionally provided to tf.function to control the graphs traced.
    # The input signature specifies the shape and type of each Tensor argument to the function using
    # a tf.TensorSpec object.


    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:  # 이 속에서는 input부터 cost까지의 forward pass를 정의한다.
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)  # this is a classification with tokens.

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)  # 그냥 loss의 mean을 계산함.
        train_accuracy(accuracy_function(tar_real, predictions))


    # Portuguese is used as the input language and English is the target language.

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:  # batch 50 마다 알려줘
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} '
                      f'Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:  # 5 epoch 마다 저장.
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


    # ## Evaluate

    # The following steps are used for evaluation:
    #
    # * Encode the input sentence using the Portuguese tokenizer (`tokenizers.pt`). This is the encoder input.
    # * The decoder input is initialized to the `[START]` token.
    # * Calculate the padding masks and the look ahead masks.
    # * The `decoder` then outputs the predictions by looking at the `encoder output` and its own output (self-attention).
    # * The model makes predictions of the next word for each word in the output. Most of these are redundant. Use the predictions from the last word.
    # * Concatenate the predicted word to the decoder input and pass it to the decoder.
    # * In this approach, the decoder predicts the next word based on the previous words it predicted.
    #
    # Note: The model used here has less capacity to keep the example relatively faster so the predictions maybe less right. To reproduce the results in the paper, use the entire dataset and base transformer model or transformer XL, by changing the hyperparameters above.

    # In[57]:


    def evaluate(sentence, max_length=40):  # predict 함수
        '''
        :param sentence: 진짜 str sentence
        :param max_length: 최대 문장길이 인듯
        :return:
        '''
        # inp sentence is portuguese, hence adding the start and end token
        sentence = tf.convert_to_tensor([sentence])
        sentence = tokenizers.pt.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # as the target is english, the first word to the transformer should be the
        # english start token.
        start, end = tokenizers.en.tokenize([''])[0]  # 빈 sentence 를 tokenize 하면 <SOS> 와 <EOS> 만 나온다.
        # 정확히는 tf.raggedtensor([<SOS>토큰, <EOS>토큰]) 이렇게 나온다.
        output = tf.convert_to_tensor([start])
        output = tf.expand_dims(output, 0)

        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == end:
                break

        # output.shape (1, tokens)
        text = tokenizers.en.detokenize(output)[0]  # shape: ()

        tokens = tokenizers.en.lookup(output)[0]

        return text, tokens, attention_weights


    # In[58]:


    def print_translation(sentence, tokens, ground_truth):
        print(f'{"Input:":15s}: {sentence}')
        print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
        print(f'{"Ground truth":15s}: {ground_truth}')


    # In[59]:


    sentence = "este é um problema que temos que resolver."
    ground_truth = "this is a problem we have to solve ."

    translated_text, translated_tokens, attention_weights = evaluate(sentence)
    print_translation(sentence, translated_text, ground_truth)

    # In[60]:


    sentence = "os meus vizinhos ouviram sobre esta ideia."
    ground_truth = "and my neighboring homes heard about this idea ."

    translated_text, translated_tokens, attention_weights = evaluate(sentence)
    print_translation(sentence, translated_text, ground_truth)

    # In[61]:


    sentence = "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram."
    ground_truth = "so i \'ll just share with you some stories very quickly of some magical things that have happened ."

    translated_text, translated_tokens, attention_weights = evaluate(sentence)
    print_translation(sentence, translated_text, ground_truth)

    # You can pass different layers and attention blocks of the decoder to the `plot` parameter.

    # ## Attention plots

    # The `evaluate` function also returns a dictionary of attention maps you can use to visualize the internal working of the model:

    # In[62]:


    sentence = "este é o primeiro livro que eu fiz."
    ground_truth = "this is the first book i've ever done."

    translated_text, translated_tokens, attention_weights = evaluate(sentence)
    print_translation(sentence, translated_text, ground_truth)


    # In[63]:


    def plot_attention_head(in_tokens, translated_tokens, attention):
        # The plot is of the attention when a token was generated.
        # The model didn't generate `<START>` in the output. Skip it.
        translated_tokens = translated_tokens[1:]

        ax = plt.gca()
        ax.matshow(attention)
        ax.set_xticks(range(len(in_tokens)))
        ax.set_yticks(range(len(translated_tokens)))

        labels = [label.decode('utf-8') for label in in_tokens.numpy()]
        ax.set_xticklabels(
            labels, rotation=90)

        labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
        ax.set_yticklabels(labels)


    # In[64]:


    head = 0
    # shape: (batch=1, num_heads, seq_len_q, seq_len_k)
    attention_heads = tf.squeeze(
        attention_weights['decoder_layer4_block2'], 0)
    attention = attention_heads[head]
    attention.shape

    # In[65]:


    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.pt.lookup(in_tokens)[0]
    in_tokens

    # In[66]:


    translated_tokens

    # In[67]:


    plot_attention_head(in_tokens, translated_tokens, attention)


    # In[68]:


    def plot_attention_weights(sentence, translated_tokens, attention_heads):
        in_tokens = tf.convert_to_tensor([sentence])
        in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
        in_tokens = tokenizers.pt.lookup(in_tokens)[0]
        in_tokens

        fig = plt.figure(figsize=(16, 8))

        for h, head in enumerate(attention_heads):
            ax = fig.add_subplot(2, 4, h + 1)

            plot_attention_head(in_tokens, translated_tokens, head)

            ax.set_xlabel(f'Head {h + 1}')

        plt.tight_layout()
        plt.show()


    # In[69]:


    plot_attention_weights(sentence, translated_tokens,
                           attention_weights['decoder_layer4_block2'][0])

    # The model does okay on unfamiliar words. Neither "triceratops" or "encyclopedia" are in the input dataset and the model almost learns to transliterate them, even without a shared vocabulary:

    # In[70]:


    sentence = "Eu li sobre triceratops na enciclopédia."
    ground_truth = "I read about triceratops in the encyclopedia."

    translated_text, translated_tokens, attention_weights = evaluate(sentence)
    print_translation(sentence, translated_text, ground_truth)

    plot_attention_weights(sentence, translated_tokens,
                           attention_weights['decoder_layer4_block2'][0])

    # ## Summary
    #
    # In this tutorial, you learned about positional encoding, multi-head attention, the importance of masking and how to create a transformer.
    #
    # Try using a different dataset to train the transformer. You can also create the base transformer or transformer XL by changing the hyperparameters above. You can also use the layers defined here to create [BERT](https://arxiv.org/abs/1810.04805) and train state of the art models. Furthermore, you can implement beam search to get better predictions.
