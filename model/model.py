from vgg import vgg16
import tensorflow as tf
from LSTM import lstm
import settings
from ACC.acc import SequenceAccuracy
from LOSS.ctc import CTCLoss
class CRNN:
    def __init__(self):
        with open(settings.table_path, 'r', encoding='utf8') as f:
            self.output_features = len(f.readlines()) + 1
        self.input_features =(settings.input_shape[0],None,settings.input_shape[2])
        self.rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1./ 255)
        self.rnn_network = lstm
        self.cnn_network = vgg16
        self.dense = tf.keras.layers.Dense(self.output_features)

    def build(self):
        inputs_img = tf.keras.Input(shape=self.input_features, name='input_data')
        x=self.rescaling(inputs_img)
        x = self.cnn_network(x)
        x = self.rnn_network(x)
        outputs = self.dense(x)
        model = tf.keras.Model(inputs=inputs_img, outputs=outputs, name="crnn")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=settings.initial_learning_rate),
            loss=CTCLoss(),
            metrics=[SequenceAccuracy()]
        )
        model.summary()
       # tf.keras.utils.plot_model(model)
        return model
if __name__ == '__main__':
    model=CRNN()
    model.build()
