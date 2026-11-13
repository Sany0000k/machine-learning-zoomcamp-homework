import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path

columns = [
    'poisonous',
    'cap-shape',
    'cap-surface',
    'cap-color',
    'bruises',
    'odor',
    'gill-attachment',
    'gill-spacing',
    'gill-size',
    'gill-color',
    'stalk-shape',
    'stalk-root',
    'stalk-surface-above-ring',
    'stalk-surface-below-ring',
    'stalk-color-above-ring',
    'stalk-color-below-ring',
    'veil-type',
    'veil-color',
    'ring-number',
    'ring-type',
    'spore-print-color',
    'population',
    'habitat'
]

best_threshold = 0.3
output_file = Path(__file__).resolve().parent / "model.bin" #'model.bin'

DATA_FILE = Path(__file__).resolve().parent / "agaricus-lepiota.data"
df = pd.read_csv(
    DATA_FILE,
    header=None,
    names=columns,
    na_values='?'
)

df['stalk-root'] = df['stalk-root'].fillna('m')
del df['veil-type']
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_full_train = df_full_train.poisonous.values
y_test = df_test.poisonous.values
del df_test['poisonous']
del df_full_train['poisonous']

def train(df_train, y_train):
    rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
    )

    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    rf.fit(X_train, y_train)
    return dv, rf
    
def predict(df, dv, model):
    dict = df.to_dict(orient='records')
    X_test = dv.transform(dict)
    p_index = list(model.classes_).index('p')
    y_test_proba = model.predict_proba(X_test)[:, p_index]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    return y_test_pred

print('training the final model')
dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, dv, model)



auc = roc_auc_score((y_test == "p").astype(int), y_pred)
print(f"auc={auc:.3f}")

# auc = roc_auc_score(y_test, y_pred)
# print(f'auc={auc}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')

