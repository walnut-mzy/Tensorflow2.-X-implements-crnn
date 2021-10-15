import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
def vgg16(inputs):
    #输入
    inputs=inputs
    #BLOCK1
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding="same", activation="relu")(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding="valid")(x)
    #print(x)
    #BLOCK2
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)
    #print(x)
    #BLOCK3
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)
    #print(x)
    #BLOCK4
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)
    #print(x)
    #BLOCK5
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)
    print(x)
    #输出层
    output=tf.keras.layers.Reshape((-1, 512))(x)
    return output

if __name__ == '__main__':

    inputs=tf.keras.layers.Input(shape=(224,112,3))
    vgg= Model(inputs=inputs, outputs=vgg16(inputs))
    vgg.summary()
    # tf.keras.utils.plot_model(vgg)  # 绘制模型图
