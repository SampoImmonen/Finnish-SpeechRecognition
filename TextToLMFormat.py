import random
import pandas as pd
from IPython.display import display, HTML
import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\...\…\–\é]'

def remove_special_characters(sent):
    sent = re.sub(chars_to_ignore_regex, '', sent).lower() + " "
    return sent

singlespeaker = "data/singlespeaker/train.csv"
commonvoice = "data/commonvoice/train.csv"
speechcollector = "data/speechcollector/train.csv"
voxpopuli = "data/fi/train.csv"


if __name__ == '__main__':


    train_dfs = [singlespeaker, commonvoice, speechcollector, voxpopuli]

    dfs = []
    for df in train_dfs:
        train_df = pd.read_csv(df)
        dfs.append(train_df)
    
    with open("data/owndata_cleaned.txt", 'w') as f:
        for df in dfs:
            sentences = df['sentence']
            for sent in sentences:
                clean_sent = remove_special_characters(sent)
                f.write(clean_sent+'\n')