from model import CRNN
import settings
import tensorflow as tf
import os
from process.dataset_process import Preprocess

def train():
    dataset=Preprocess()
    train_dataset=dataset.build("train")
    val_dataset= dataset.build('val')
    train_dataset_size=dataset.size('train')
    val_dataset_size=dataset.size('val')
    model=CRNN()
    model = model.build()
    model.load_weights(os.path.join(settings.save_path, 'crnn_{0}.h5'.format(str(settings.initial_epoch))))
    callbacks = [
        # 模型保存
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(settings.save_path, "crnn_{epoch}.h5"),
            monitor='val_loss',
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=20,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)
    ]
    # 查看模型结构
    model.summary()
    model.fit(
        train_dataset,
        epochs=settings.epoch,
        steps_per_epoch=train_dataset_size // settings.batch,
        initial_epoch=settings.initial_epoch,
        validation_data=val_dataset,
        validation_steps=val_dataset_size // settings.batch,
        callbacks=callbacks,
    )
