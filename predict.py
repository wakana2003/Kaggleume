import pandas as pd

# ファイルのパス
file_paths = {
    "item_categories": "mnt\data\item_categories.csv",
    "items": "mnt\data\items.csv",
    "sales_train": "mnt\data\sales_train.csv",
    "sample_submission": "mnt\data\sample_submission.csv",
    "shops": "mnt\data\shops.csv",
    "test": "mnt\data\test.csv",
}

# 各データの読み込み
data = {name: pd.read_csv(path) for name, path in file_paths.items()}

# 各データの基本情報を確認
data_info = {name: df.info() for name, df in data.items()}

# 各データの最初の数行を表示
data_head = {name: df.head() for name, df in data.items()}

data_head

# データのクリーニング

# 1. `item_cnt_day` の異常値（負の値や極端な値）を処理
data["sales_train"] = data["sales_train"][data["sales_train"]["item_cnt_day"].between(0, 20)]

# 2. `item_price` の異常値を処理（例えば、0以下の価格や極端に高い値を除外）
data["sales_train"] = data["sales_train"][data["sales_train"]["item_price"] > 0]

# 3. `test.csv` に存在しない `shop_id` / `item_id` のデータを削除
valid_shops = data["test"]["shop_id"].unique()
valid_items = data["test"]["item_id"].unique()

data["sales_train"] = data["sales_train"][data["sales_train"]["shop_id"].isin(valid_shops)]
data["sales_train"] = data["sales_train"][data["sales_train"]["item_id"].isin(valid_items)]

# クリーニング後のデータ確認
data["sales_train"].describe()


# 月ごとの販売数 (`item_cnt_month`) を集計
monthly_sales = data["sales_train"].groupby(["date_block_num", "shop_id", "item_id"])["item_cnt_day"].sum().reset_index()
monthly_sales.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)

# 販売数の範囲を [0, 20] にクリップ（評価指標に合わせる）
monthly_sales["item_cnt_month"] = monthly_sales["item_cnt_month"].clip(0, 20)

# データの確認
import ace_tools as tools
tools.display_dataframe_to_user(name="Monthly Sales Data", dataframe=monthly_sales)

# ラグ特徴量の作成関数
def create_lag_features(df, lags, column):
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df.groupby(["shop_id", "item_id"])[column].shift(lag)
    return df

# 1, 2, 3, 6, 12ヶ月のラグを追加
lags = [1, 2, 3, 6, 12]
monthly_sales = create_lag_features(monthly_sales, lags, "item_cnt_month")

# データの確認
tools.display_dataframe_to_user(name="Lag Features Data", dataframe=monthly_sales)

# ショップごとの平均販売数
shop_mean_sales = monthly_sales.groupby(["date_block_num", "shop_id"])["item_cnt_month"].mean().reset_index()
shop_mean_sales.rename(columns={"item_cnt_month": "mean_shop_sales"}, inplace=True)
monthly_sales = monthly_sales.merge(shop_mean_sales, on=["date_block_num", "shop_id"], how="left")

# アイテムごとの平均販売数
item_mean_sales = monthly_sales.groupby(["date_block_num", "item_id"])["item_cnt_month"].mean().reset_index()
item_mean_sales.rename(columns={"item_cnt_month": "mean_item_sales"}, inplace=True)
monthly_sales = monthly_sales.merge(item_mean_sales, on=["date_block_num", "item_id"], how="left")

# カテゴリごとの平均販売数
item_category_mapping = data["items"][["item_id", "item_category_id"]]
monthly_sales = monthly_sales.merge(item_category_mapping, on="item_id", how="left")

category_mean_sales = monthly_sales.groupby(["date_block_num", "item_category_id"])["item_cnt_month"].mean().reset_index()
category_mean_sales.rename(columns={"item_cnt_month": "mean_category_sales"}, inplace=True)
monthly_sales = monthly_sales.merge(category_mean_sales, on=["date_block_num", "item_category_id"], how="left")

# データの確認
tools.display_dataframe_to_user(name="Sales Data with Aggregated Features", dataframe=monthly_sales)

# アイテムごとの価格統計量を計算
item_price_stats = data["sales_train"].groupby("item_id")["item_price"].agg(["mean", "median", "std"]).reset_index()
item_price_stats.rename(columns={"mean": "mean_item_price", "median": "median_item_price", "std": "std_item_price"}, inplace=True)

# 特徴量データにマージ
monthly_sales = monthly_sales.merge(item_price_stats, on="item_id", how="left")

# データの確認
tools.display_dataframe_to_user(name="Sales Data with Price Features", dataframe=monthly_sales)

# アイテムごとの価格統計量を計算
item_price_stats = data["sales_train"].groupby("item_id")["item_price"].agg(["mean", "median", "std"]).reset_index()
item_price_stats.rename(columns={"mean": "mean_item_price", "median": "median_item_price", "std": "std_item_price"}, inplace=True)

# 特徴量データにマージ
monthly_sales = monthly_sales.merge(item_price_stats, on="item_id", how="left")

# データの確認
tools.display_dataframe_to_user(name="Sales Data with Price Features", dataframe=monthly_sales)

