import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import f1_score

# データの読み込み
train = pd.read_csv('./source/train.csv', index_col=0)
test = pd.read_csv('./source/test.csv', index_col=0)
sample_submit = pd.read_csv('./source/sample_submission.csv', index_col=0, header=None)

# 欠損値の処理
train.fillna('NULL', inplace=True)
test.fillna('NULL', inplace=True)

# problems列の処理
train['bool_problems'] = train['problems'].apply(lambda x: 0 if x=='NULL' else 1)
test['bool_problems'] = test['problems'].apply(lambda x: 0 if x=='NULL' else 1)

# 学習用データと検証用データの分割
train, valid = train_test_split(train, test_size=0.2, stratify=train['health'], random_state=82)

# 使用する特徴量の選択
select_cols = ['tree_dbh', 'curb_loc', 'sidewalk', 'steward', 'guards', 'user_type', 'bool_problems']

# 目的変数とそれ以外に学習用データを分割
x_train = train[select_cols]
y_train = train['health']
x_valid = valid[select_cols]
y_valid = valid['health']

# カテゴリのままでは学習できないのでワンホットエンコーディングで数値化
x_train = pd.get_dummies(x_train)
x_valid = pd.get_dummies(x_valid)
test = pd.get_dummies(test[select_cols])

# AutoML（TPOT）を使用
tpot = TPOTClassifier(generations=5, population_size=20, random_state=42, verbosity=2, n_jobs=-1)
tpot.fit(x_train, y_train)

# 検証データでの評価
valid_predictions = tpot.predict(x_valid)
valid_f1 = f1_score(y_valid, valid_predictions, average='macro')
print(f"Validation F1 Score (Macro): {valid_f1}")

# 予測
pred = tpot.predict(test)

# 予測結果の保存
sample_submit[1] = pred
sample_submit.to_csv('./submit/submit_automl_v1.csv', header=None)
