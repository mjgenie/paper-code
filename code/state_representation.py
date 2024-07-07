import tensorflow as tf


class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()


    def call(self, x):
        #x[1]: item embedding
        items_eb = tf.transpose(x[1], perm=(0, 2, 1)) / self.embedding_dim
        #1D convolution layer 적용
        wav = self.wav(items_eb)
        wav = tf.transpose(wav, perm=(0, 2, 1))
        wav = tf.squeeze(wav, axis=1)
        #x[0]과 wav 텐서 간의 요소별 dot product
        user_wav = tf.keras.layers.multiply([x[0], wav])
        #user embedding인 x[0], user_wav, wav 결합
        concat = self.concat([x[0], user_wav, wav])
        return self.flatten(concat)