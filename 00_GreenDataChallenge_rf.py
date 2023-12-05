import pandas as pd

# データの読み込み
train = pd.read_csv('./source/train.csv', index_col=0)
test = pd.read_csv('./source/test.csv', index_col=0)
sample_submit = pd.read_csv('./source/sample_submission.csv', index_col=0, header=None)

# 欠損値の処理
train.fillna('NULL', inplace=True)

# problems列の処理
train['bool_problems'] = train['problems'].apply(lambda x: 0 if x=='NULL' else 1)

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

# 木の直径と健康状態の関係を可視化
plt.figure(figsize=(10, 6))
sns.boxplot(x='health', y='tree_dbh', data=train)
plt.title('Relationship between "tree_dbh" and "health"')
plt.show()

# カテゴリ変数と健康状態の関係を可視化
categorical_features = ['curb_loc', 'sidewalk', 'steward', 'guards', 'user_type', 'bool_problems']

fig, axes = plt.subplots(len(categorical_features), 1, figsize=(10, 20))

for i, feature in enumerate(categorical_features):
    sns.countplot(x='health', hue=feature, data=train, ax=axes[i])
    axes[i].set_title(f'Health Status by {feature}')
    axes[i].legend(title=feature, loc='upper right')

plt.tight_layout()
plt.show()

# モデリングの準備
test.fillna('NULL', inplace=True)
test['bool_problems'] = test['problems'].apply(lambda x: 0 if x=='NULL' else 1)

# 学習用データと検証用データの分割
from sklearn.model_selection import train_test_split
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

# モデリング
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

model = RandomForestClassifier()
model.fit(x_train, y_train)

# 検証データでの評価
valid_predictions = model.predict(x_valid)
valid_f1 = f1_score(y_valid, valid_predictions, average='macro')
print(f"Validation F1 Score (Macro): {valid_f1}")

# 予測
pred = model.predict(test)

# 予測結果の保存
sample_submit[1] = pred
sample_submit.to_csv('./submit/submit_sample.csv', header=None)