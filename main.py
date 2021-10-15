from train import train
import settings
from predict import predict
mode=settings.mode

if mode=="train":
    train()
elif mode=="predict":
    predict()