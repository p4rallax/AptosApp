import pandas as pd 
from sklearn.model_selection import train_test_split, StratifiedKFold

train_csv = pd.read_csv('../content/train/train.csv')
train_df, val_df = train_test_split(train_csv, test_size=0.1, random_state=2018, stratify=train_csv.diagnosis)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
train_df.to_csv('train_split.csv')
val_df.to_csv('val_split.csv')