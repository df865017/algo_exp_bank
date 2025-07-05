from typing import Any

import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Identity, Layer, LayerNormalization
from tensorflow import Tensor, TensorShape

import logging
from IPython import display

from model.tensorflow.embedding import Embedding

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

# from model.tensorflow.mlp import MLP
class MLP(Layer):
    def __init__(
        self,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float,
        kernel_regularizer=None,  # 新增正则化参数
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer  # 存储正则化器

        self.layers = []
        for _ in range(num_hidden):
            self.layers.append(
                tf.keras.layers.Dense(
                    dim_hidden,
                    activation="relu",
                    kernel_regularizer=kernel_regularizer  # 应用正则化
                )
            )
            self.layers.append(tf.keras.layers.Dropout(dropout))

        self.output_layer = tf.keras.layers.Dense(
            dim_out,
            activation=None,
            kernel_regularizer=kernel_regularizer  # 应用正则化
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_hidden": self.num_hidden,
            "dim_hidden": self.dim_hidden,
            "dim_out": self.dim_out,
            "dropout": self.dropout,
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

class LinearCompressBlock(Layer):
    def __init__(
        self,
        num_emb_out: int,
        weights_initializer: str = "he_uniform",
        kernel_regularizer=None,  # 新增 L2 正则化参数
        name: str = "lcb"
    ) -> None:
        super().__init__(name=name)
        self.num_emb_out = num_emb_out
        self.weights_initializer = weights_initializer
        self.kernel_regularizer = kernel_regularizer  # 存储正则化器

    def build(self, input_shape: TensorShape) -> None:
        num_emb_in = input_shape[1]

        self.weight = self.add_weight(
            name="weight",
            shape=(num_emb_in, self.num_emb_out),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
            regularizer=self.kernel_regularizer  # 应用 L2 正则化
        )
        self.built = True

    def call(self, inputs: Tensor) -> Tensor:
        logging.debug("f02-01-01:inputs : {}".format(inputs))
        outputs = tf.transpose(inputs, (0, 2, 1))
        logging.debug("f02-01-02:outputs : {}".format(outputs))
        outputs = outputs @ self.weight
        logging.debug("f02-01-03:outputs : {}".format(outputs))
        outputs = tf.transpose(outputs, (0, 2, 1))
        logging.debug("f02-01-04:outputs : {}".format(outputs))
        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_emb_out": self.num_emb_out,
                "weights_initializer": self.weights_initializer,
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),  # 序列化正则化器
            }
        )
        return config


class FactorizationMachineBlock(Layer):
    def __init__(
        self,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        weights_initializer: str = "he_uniform",
        kernel_regularizer=None,  # 新增 L2 正则化参数
        name: str = "fmb",
    ) -> None:
        super().__init__(name=name)

        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank
        self.weights_initializer = weights_initializer
        self.kernel_regularizer = kernel_regularizer  # 存储正则化器

        self.norm = LayerNormalization()
        self.mlp = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer  # 传递到 MLP
        )

    def build(self, input_shape: TensorShape) -> None:
        self.num_emb_in = input_shape[1]
        print("input_shape:{}",input_shape)

        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_emb_in, self.rank),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
            regularizer=self.kernel_regularizer  # 应用 L2 正则化
        )
        self.built = True

    def call(self, inputs: Tensor) -> Tensor:
        logging.debug("f02-02-01:inputs : {}".format(inputs))
        outputs = tf.transpose(inputs, (0, 2, 1))
        logging.debug("f02-02-02:outputs : {}".format(outputs))
        outputs = outputs @ self.weight

        logging.debug("f02-02-03:outputs : {}".format(outputs))
        outputs = inputs @ outputs
        logging.debug("f02-02-03-01:outputs : {}".format(outputs))

        outputs = tf.reshape(outputs, (-1, self.num_emb_in * self.rank))
        logging.debug("f02-02-03-02:num_emb_in : {}".format(self.num_emb_in))
        logging.debug("f02-02-03-03:rank : {}".format(self.rank))
        logging.debug("f02-02-04:outputs : {}".format(outputs))

        logging.debug("f02-02-05:outputs : {}".format(outputs))
        outputs = self.mlp(self.norm(outputs))
        logging.debug("f02-02-06:outputs : {}".format(outputs))
        outputs = tf.reshape(outputs, (-1, self.num_emb_out, self.dim_emb))
        logging.debug("f02-02-07:outputs : {}".format(outputs))
        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_emb_out": self.num_emb_out,
                "dim_emb": self.dim_emb,
                "rank": self.rank,
                "weights_initializer": self.weights_initializer,
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),  # 序列化正则化器
            }
        )
        return config


class ResidualProjection(Layer):
    def __init__(
        self,
        num_emb_out: int,
        weights_initializer: str = "he_uniform",
        kernel_regularizer=None,  # 新增 L2 正则化参数
        name: str = "residual_projection"
    ) -> None:
        super().__init__(name=name)

        self.num_emb_out = num_emb_out
        self.weights_initializer = weights_initializer
        self.kernel_regularizer = kernel_regularizer  # 存储正则化器

    def build(self, input_shape: TensorShape) -> None:
        self.num_emb_in = input_shape[1]

        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_emb_in, self.num_emb_out),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
            regularizer=self.kernel_regularizer  # 应用 L2 正则化
        )
        self.built = True

    def call(self, inputs: Tensor) -> Tensor:
        logging.debug("f02-04-01:inputs : {}".format(inputs))
        outputs = tf.transpose(inputs, (0, 2, 1))
        logging.debug("f02-04-02:outputs : {}".format(outputs))
        outputs = outputs @ self.weight
        logging.debug("f02-04-03:outputs : {}".format(outputs))
        outputs = tf.transpose(outputs, (0, 2, 1))
        logging.debug("f02-04-04:outputs : {}".format(outputs))
        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_emb_out": self.num_emb_out,
                "weights_initializer": self.weights_initializer,
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),  # 序列化正则化器
            }
        )
        return config


class WukongLayer(Layer):
    def __init__(
        self,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        kernel_regularizer=None,  # 新增 L2 正则化参数
        name: str = "wukong",
    ) -> None:
        super().__init__(name=name)

        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.rank_fmb = rank_fmb
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer  # 存储正则化器
        self.norm = LayerNormalization()

    def build(self, input_shape: TensorShape) -> None:
        num_emb_in, dim_emb = input_shape[-2:]

        print("dim_emb ****:",dim_emb)

        self.lcb = LinearCompressBlock(
            self.num_emb_lcb,
            kernel_regularizer=self.kernel_regularizer  # 传递到 LCB
        )
        self.fmb = FactorizationMachineBlock(
            self.num_emb_fmb,
            dim_emb,
            self.rank_fmb,
            self.num_hidden,
            self.dim_hidden,
            self.dropout,
            kernel_regularizer=self.kernel_regularizer  # 传递到 FMB
        )

        if num_emb_in != self.num_emb_lcb + self.num_emb_fmb:
            self.residual_projection = ResidualProjection(
                self.num_emb_lcb + self.num_emb_fmb,
                kernel_regularizer=self.kernel_regularizer  # 传递到 ResidualProjection
            )
        else:
            self.residual_projection = Identity()

        self.built = True

    def call(self, inputs: Tensor) -> Tensor:
        logging.debug("f02-01:inputs : {}".format(inputs))
        lcb = self.lcb(inputs)
        logging.debug("f02-02:lcb : {}".format(lcb))
        fmb = self.fmb(inputs)
        logging.debug("f02-03:fmb : {}".format(fmb))
        outputs = tf.concat((fmb, lcb), axis=1)
        residual_res = self.residual_projection(inputs)
        logging.debug("f02-05-01:outputs : {}".format(outputs))
        logging.debug("f02-05-02:residual_res : {}".format(residual_res))
        outputs = self.norm(outputs + residual_res)
        logging.debug("f02-05:outputs : {}".format(outputs))

        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_emb_lcb": self.num_emb_lcb,
                "num_emb_fmb": self.num_emb_fmb,
                "rank_fmb": self.rank_fmb,
                "num_hidden": self.num_hidden,
                "dim_hidden": self.dim_hidden,
                "dropout": self.dropout,
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),  # 序列化正则化器
            }
        )
        return config


class Wukong(Model):
    def __init__(
        self,
        num_layers: int,
        num_sparse_emb: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden_wukong: int,
        dim_hidden_wukong: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        kernel_regularizer=None,  # 新增 L2 正则化参数
    ) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb

        self.embedding = Embedding(num_sparse_emb, dim_emb)

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
                    name=f"wukong_{i}",
                    kernel_regularizer=kernel_regularizer  # 传递到 WukongLayer
                ),
            )

        self.projection_head = MLP(
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            kernel_regularizer=kernel_regularizer  # 传递到 MLP
        )

    def call(self, inputs: list[Tensor]) -> Tensor:
        outputs = self.embedding(inputs)
        outputs = self.interaction_layers(outputs)
        outputs = tf.reshape(outputs, (-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb))
        outputs = self.projection_head(outputs)
        return outputs