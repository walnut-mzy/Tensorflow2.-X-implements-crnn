from model import CRNN
import os
import settings
model=CRNN()
model=model.build()
model.load_weights(os.path.join(settings.save_path+'crnn_{0}.h5'.format(str(settings.test_epoch))))
model.summary()
# 模型保存
model.save(settings.transform_model_save_path)
print('success save')