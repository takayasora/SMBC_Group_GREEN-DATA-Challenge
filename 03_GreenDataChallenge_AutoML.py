import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import f1_score

# データの読み込み
train = pd.read_csv('./source/train_enc.csv', index_col=0)
test = pd.read_csv('./source/test_enc.csv', index_col=0)
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
# 使用する特徴量の選択
select_cols = [
    'tree_dbh',  # 木の直径
    'curb_loc = OnCurb',  # 車道上にあるかどうか
    'curb_loc = OffsetFromCurb',  # 車道からのオフセット
    'health',  # 木の健康状態
    'steward = 3or4',  # 世話をしている人が3人以上の場合
    'steward = 1or2',  # 世話をしている人が1人または2人の場合
    'steward = 4orMore',  # 世話をしている人が4人以上の場合
    'guards = Helpful',  # ガードが役立つ場合
    'guards = Harmful',  # ガードが有害な場合
    'guards = Unsure',  # ガードの効果が分からない場合
    'sidewalk = Damage',  # 舗装に損傷がある場合
    'sidewalk = NoDamage',  # 舗装に損傷がない場合
    'user_type = Volunteer',  # 利用者がボランティアの場合
    'user_type = NYC Parks Staff',  # 利用者がNYC Parks Staffの場合
    'user_type = TreesCount Staff',  # 利用者がTreesCount Staffの場合
    'problems',  # 問題の有無
    'spc_common',  # 一般的な木の名前
    'spc_latin',  # 学名
    'nta',  # 近隣地域のコード
    'nta_name',  # 近隣地域の名前
    'borocode',  # 行政区のコード
    'boro_ct',  # 行政区とブロックの組み合わせコード
    'boroname = Queens',  # クイーンズ行政区に所在するかどうか
    'boroname = Bronx',  # ブロンクス行政区に所在するかどうか
    'boroname = Staten Island',  # スタテンアイランド行政区に所在するかどうか
    'boroname = Manhattan',  # マンハッタン行政区に所在するかどうか
    'boroname = Brooklyn',  # ブルックリン行政区に所在するかどうか
    'zip_city',  # 郵便番号または都市
    'cb_num',  # 地域委員会の番号
    'st_senate',  # 州上院選挙区
    'st_assem',  # 州下院選挙区
    'cncldist'  # 地方区分
]


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
sample_submit.to_csv('./submit/submit_automl_v3.csv', header=None)
