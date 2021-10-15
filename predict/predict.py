from model import CRNN
from process import dataset_process
import settings
from process.decoder import Decoder
from ACC.acc import acc
def predict():
    dataset=dataset_process.Preprocess()
    model=CRNN()
    model=model.build()
    model.load_weights(settings.weight_path)
    test_data_generator, test_labels = dataset.build_test()
    # 预测
    test_data = next(iter(test_data_generator))
    result = model.predict(test_data)

    decoder = Decoder()
    y_pred = decoder.decode(result, method='greedy')
    for i, sentense in enumerate(y_pred):
        if test_labels[i] != sentense:
            print('真实标签：{0} \t 预测结果： {1}'.format(test_labels[i], sentense))

    # 准确率
    acc1, acc2 = acc(y_pred, test_labels)
    print("========================================")
    print("严格准确率：{0} \t 相似准确率：{1}".format(acc1, acc2))