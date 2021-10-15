
import tensorflow as tf
from tensorflow import keras
import difflib

class SequenceAccuracy(keras.metrics.Metric):
    def __init__(self, name='sequence_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_shape = tf.shape(y_true)
        batch_size = y_true_shape[0]
        y_pred_shape = tf.shape(y_pred)
        max_width = tf.maximum(y_true_shape[1], y_pred_shape[1])
        logit_length = tf.fill([batch_size], y_pred_shape[1])
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        y_true = self.to_dense(y_true, [batch_size, max_width])
        y_pred = self.to_dense(decoded[0], [batch_size, max_width])
        num_errors = tf.math.reduce_any(
            tf.math.not_equal(y_true, y_pred), axis=1)
        num_errors = tf.cast(num_errors, tf.float32)
        num_errors = tf.reduce_sum(num_errors)
        batch_size = tf.cast(batch_size, tf.float32)
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)

    def to_dense(self, tensor, shape):
        tensor = tf.sparse.reset_shape(tensor, shape)
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor = tf.cast(tensor, tf.float32)
        return tensor

    def result(self):
        return self.count / tf.clip_by_value(self.total, 1e-8, tf.reduce_max(self.total))

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)

def acc(label_list, result_list):
    """
    计算准确率
    严格准确率、相似准确率
    """
    flag_1 = 0
    flag_2 = 0
    for i in range(len(label_list)):
        if label_list[i] == result_list[i]:
            flag_1 += 1
        flag_2 += difflib.SequenceMatcher(None, label_list[i], result_list[i]).quick_ratio()
    return flag_1/len(label_list), flag_2/len(label_list)