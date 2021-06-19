"""
evaluation script for trained Speechrecognition models
metrics:
    WER
    CER
with and without language model

"""
import pandas as pd
from SpeechRecognizer import *

test1 = "data/commonvoice/test.csv"
test2 = "data/eduskunnanpuheet/uudetpuheet/dev-eval/test.csv"

test_df = pd.concat([pd.read_csv(test1), pd.read_csv(test2)])

SpeechRecognizer(model_dir="best_model/")
decoder