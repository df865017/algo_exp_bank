
# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.regularizers import l2
import logging
from IPython import display

class Identity(Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

# 自定义日志处理器
class DisplayLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        formatted_msg = "**{}**".format(msg)
        display.display(display.Markdown(formatted_msg))

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = DisplayLogHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ============ 安全获取形状 ============
def get_shape_dim(shape, index):
    if isinstance(shape, tf.TensorShape):
        return shape[index].value if shape[index] is not None else None
    elif isinstance(shape, tf.Dimension):
        return shape.value
    else:
        return shape

# ============ 自定义 LayerNormalization ============
class LayerNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = get_shape_dim(input_shape, -1)
        if dim is None:
            raise ValueError("输入形状的最后一个维度必须是静态值")
        self.gamma = self.add_weight(
            name='gamma',
            shape=(dim,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(dim,),
            initializer='zeros',
            trainable=True
        )
        self.built = True

    def call(self, inputs, training=None):
        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

# ============ 自定义层 ============

class LinearCompressBlock(Layer):
    def __init__(self, num_emb_out, weights_initializer='he_uniform', kernel_regularizer=None, **kwargs):
        super(LinearCompressBlock, self).__init__(**kwargs)
        self.num_emb_out = num_emb_out
        self.weights_initializer = VarianceScaling(
            scale=2.0, mode='fan_in', distribution='uniform'
        ) if weights_initializer == 'he_uniform' else weights_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        num_emb_in = get_shape_dim(input_shape, 1)
        if num_emb_in is None:
            raise ValueError("输入形状的第二个维度必须是静态值")
        self.weight = self.add_weight(
            name='weight',
            shape=(num_emb_in, self.num_emb_out),
            initializer=self.weights_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True
        )
        self.built = True

    def call(self, inputs):
        logging.debug("f02-01-01:inputs : {}".format(inputs))
        outputs = tf.transpose(inputs, (0, 2, 1))  # (batch_size, dim_emb, num_emb_in)
        logging.debug("f02-01-02:outputs : {}".format(outputs))

        # 使用 tf.tensordot 替代 tf.matmul，支持 3D × 2D 的矩阵乘法
        # 轴 2 (dim_emb, num_emb_in) × 轴 0 (num_emb_in, num_emb_out)
        outputs = tf.tensordot(outputs, self.weight, axes=[[2], [0]])  # (batch_size, dim_emb, num_emb_out)
        logging.debug("f02-01-03:outputs : {}".format(outputs))
        
        outputs = tf.transpose(outputs, (0, 2, 1))  # (batch_size, num_emb_out, dim_emb)
        logging.debug("f02-01-04:outputs : {}".format(outputs))
        return outputs

    def compute_output_shape(self, input_shape):
        batch_size = get_shape_dim(input_shape, 0)
        dim_emb = get_shape_dim(input_shape, 2)
        return (batch_size, self.num_emb_out, dim_emb)

    
class FactorizationMachineBlock(Layer):
    def __init__(self, num_emb_out, dim_emb, rank, num_hidden, dim_hidden, dropout,
                 weights_initializer='he_uniform', kernel_regularizer=None, **kwargs):
        super(FactorizationMachineBlock, self).__init__(**kwargs)
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.weights_initializer = VarianceScaling(
            scale=2.0, mode='fan_in', distribution='uniform'
        ) if weights_initializer == 'he_uniform' else weights_initializer
        self.kernel_regularizer = kernel_regularizer
        self.norm = LayerNormalization()

    def build(self, input_shape):
        num_emb_in = get_shape_dim(input_shape, 1)
        self.num_emb_in = num_emb_in
        if num_emb_in is None:
            raise ValueError("输入形状的第二个维度必须是静态值")
#         self.dim_emb = 128
        self.dim_emb = 51

        self.weight = self.add_weight(
            name='weight',
            shape=(num_emb_in, self.rank),
            initializer=self.weights_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True
        )

        # 在 build 方法中初始化 MLP，确保输入形状已确定
        self.mlp = MLP(
            num_hidden=self.num_hidden,
            dim_hidden=self.dim_hidden,
            dim_out=self.num_emb_out * self.dim_emb,
            dropout=self.dropout,
            kernel_regularizer=self.kernel_regularizer
        )
        self.built = True

    def _get_static_dim(self, dim):
        if dim is None:
            raise ValueError("维度必须是静态值")
        return int(dim)

    def call(self, inputs):
        logging.debug("f02-02-01:inputs : {}".format(inputs))
        outputs = tf.transpose(inputs, (0, 2, 1))  # (batch, dim, num_emb_in)
        logging.debug("f02-02-02:outputs : {}".format(outputs))
        outputs = tf.tensordot(outputs, self.weight, axes=[[2], [0]])  # (batch_size, dim_emb, num_emb_out)

        logging.debug("f02-02-03:outputs : {}".format(outputs))
#         outputs = tf.tensordot(inputs, outputs, axes=[[2], [1]])  # (batch, num_emb_in, rank)
#         outputs = tf.reduce_sum(outputs, axis=2)   # (batch, num_emb_in, rank)
        outputs = tf.einsum('bij,bjk->bik', inputs, outputs)
        logging.debug("f02-02-03-01:outputs : {}".format(outputs))
        
        outputs = tf.reshape(outputs, (-1, self.num_emb_in * self.rank))
        logging.debug("f02-02-03-02:num_emb_in : {}".format(self.num_emb_in))
        logging.debug("f02-02-03-03:rank : {}".format(self.rank))
        logging.debug("f02-02-04:outputs : {}".format(outputs))
        
#         outputs = tf.reshape(outputs, (-1, self.num_emb_in * self.rank))  # (batch, num_emb_in * rank)
        logging.debug("f02-02-05:outputs : {}".format(outputs))
        outputs = self.mlp(self.norm(outputs))     # (batch, num_emb_out * dim_emb)
        logging.debug("f02-02-06:outputs : {}".format(outputs))
        outputs = tf.reshape(outputs, (-1, self.num_emb_out, self.dim_emb))  # (batch, num_emb_out, dim_emb)
        logging.debug("f02-02-07:outputs : {}".format(outputs))
        return outputs

    def compute_output_shape(self, input_shape):
        batch_size = get_shape_dim(input_shape, 0)
        return (batch_size, self.num_emb_out, self.dim_emb)



class ResidualProjection(Layer):
    def __init__(self, num_emb_out, weights_initializer='he_uniform', kernel_regularizer=None, **kwargs):
        super(ResidualProjection, self).__init__(**kwargs)
        self.num_emb_out = num_emb_out
        self.weights_initializer = VarianceScaling(
            scale=2.0, mode='fan_in', distribution='uniform'
        ) if weights_initializer == 'he_uniform' else weights_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        num_emb_in = get_shape_dim(input_shape, 1)
        if num_emb_in is None:
            raise ValueError("输入形状的第二个维度必须是静态值")
        self.weight = self.add_weight(
            name='weight',
            shape=(num_emb_in, self.num_emb_out),
            initializer=self.weights_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True
        )
        self.built = True

    def call(self, inputs):
        logging.debug("f02-04-01:inputs : {}".format(inputs))
        outputs = tf.transpose(inputs, (0, 2, 1))
        logging.debug("f02-04-02:outputs : {}".format(outputs))
        
#         outputs = tf.matmul(outputs, self.weight)
        outputs = tf.tensordot(outputs, self.weight, axes=[[2], [0]])  # (batch_size, dim_emb, num_emb_out)
        logging.debug("f02-04-03:outputs : {}".format(outputs))
        outputs = tf.transpose(outputs, (0, 2, 1))
        logging.debug("f02-04-04:outputs : {}".format(outputs))
        return outputs

    def compute_output_shape(self, input_shape):
        batch_size = get_shape_dim(input_shape, 0)
        dim_emb = get_shape_dim(input_shape, 2)
        return (batch_size, self.num_emb_out, dim_emb)


class MLP(Layer):
    def __init__(self, num_hidden, dim_hidden, dim_out, dropout, kernel_regularizer=None, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.layers = []
        for _ in range(self.num_hidden):
            self.layers.append(
                Dense(self.dim_hidden, activation='relu', kernel_regularizer=self.kernel_regularizer)
            )
            self.layers.append(tf.keras.layers.Dropout(self.dropout))
        self.output_layer = Dense(self.dim_out, activation=None, kernel_regularizer=self.kernel_regularizer)
        self.built = True

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

    def compute_output_shape(self, input_shape):
        batch_size = get_shape_dim(input_shape, 0)
        return (batch_size, self.dim_out)


class WukongLayer(Layer):
    def __init__(self, num_emb_lcb, num_emb_fmb, rank_fmb, num_hidden, dim_hidden, dropout,
                 kernel_regularizer=None, **kwargs):
        super(WukongLayer, self).__init__(**kwargs)
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.rank_fmb = rank_fmb
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer
        self.norm = LayerNormalization()

    def build(self, input_shape):
        num_emb_in = get_shape_dim(input_shape, 1)
        dim_emb = get_shape_dim(input_shape, 2)
        if num_emb_in is None or dim_emb is None:
            raise ValueError("输入形状的维度必须是静态值")
        self.lcb = LinearCompressBlock(self.num_emb_lcb, kernel_regularizer=self.kernel_regularizer)
        self.fmb = FactorizationMachineBlock(
            self.num_emb_fmb,
            dim_emb,
            self.rank_fmb,
            self.num_hidden,
            self.dim_hidden,
            self.dropout,
            kernel_regularizer=self.kernel_regularizer
        )

        if num_emb_in != self.num_emb_lcb + self.num_emb_fmb:
            self.residual_projection = ResidualProjection(self.num_emb_lcb + self.num_emb_fmb, kernel_regularizer=self.kernel_regularizer)
        else:
            self.residual_projection = Identity()  # 使用 Identity 层替代 tf.identity

        self.built = True

    def call(self, inputs):
        logging.debug("f02-01:inputs : {}".format(inputs))
        lcb = self.lcb(inputs)
        logging.debug("f02-02:lcb : {}".format(lcb))
        fmb = self.fmb(inputs)
        logging.debug("f02-03:fmb : {}".format(fmb))
        outputs = tf.concat([fmb, lcb], axis=1)
        logging.debug("f02-04:outputs : {}".format(outputs))
        residual_res = self.residual_projection(inputs)
        logging.debug("f02-05-01:outputs : {}".format(outputs))
        logging.debug("f02-05-02:residual_res : {}".format(residual_res))
        outputs = self.norm(outputs + residual_res)
        logging.debug("f02-05:outputs : {}".format(outputs))

        return outputs

    def compute_output_shape(self, input_shape):
        num_emb_in = get_shape_dim(input_shape, 1)
        dim_emb = get_shape_dim(input_shape, 2)
        num_emb_out = self.num_emb_lcb + self.num_emb_fmb
        batch_size = get_shape_dim(input_shape, 0)
        return (batch_size, num_emb_out, dim_emb)


class Wukong(Model):
    def __init__(self, num_layers, num_emb_in, dim_emb, num_emb_lcb, num_emb_fmb,
                 rank_fmb, num_hidden_wukong, dim_hidden_wukong, num_hidden_head,
                 dim_hidden_head, dim_output, dropout=0.0, kernel_regularizer=None, **kwargs):
        super(Wukong, self).__init__(**kwargs)

#         # 强制约束 num_emb_in 与 num_emb_lcb + num_emb_fmb 一致
#         assert num_emb_in == num_emb_lcb + num_emb_fmb, "num_emb_in 必须等于 num_emb_lcb + num_emb_fmb"

        self.num_emb_in = num_emb_in  # 输入特征数量
        self.dim_emb = dim_emb  # 每个特征的嵌入维度
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb

        # 使用 Dense 层对浮点输入进行变换
        self.dense = Dense(self.num_emb_in * self.dim_emb, activation='linear')

        self.interaction_layers = Sequential()
        for i in range(num_layers):
            self.interaction_layers.add(
                WukongLayer(
                    num_emb_lcb,
                    num_emb_fmb,
                    rank_fmb,
                    num_hidden_wukong,
                    dim_hidden_wukong,
                    dropout,
                    kernel_regularizer=kernel_regularizer,
                    name="wukong_{}".format(i)
                )
            )

        self.projection_head = MLP(
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            kernel_regularizer=kernel_regularizer
        )

    def call(self, inputs):
        # 假设 inputs 形状为 (batch_size, num_emb_in)
        logging.debug("f01:inputs : {}".format(inputs))
        outputs = self.dense(inputs)  # 线性变换到 (batch_size, num_emb_in * dim_emb)
        outputs = tf.reshape(outputs, (-1, self.num_emb_in, self.dim_emb))  # reshape 成 (batch_size, num_emb_in, dim_emb)
        logging.debug("f02:outputs : {}".format(outputs))
        outputs = self.interaction_layers(outputs)
        logging.debug("f03:outputs : {}".format(outputs))
        outputs = tf.reshape(outputs, (-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb))
        logging.debug("f04:outputs : {}".format(outputs))
        outputs = self.projection_head(outputs)
        logging.debug("f05:outputs : {}".format(outputs))
        return outputs

    def compute_output_shape(self, input_shape):
        batch_size = get_shape_dim(input_shape, 0)
        return (batch_size, 1)
