# Daytrade Desk Trainer

デイトレードの勉強用に作った、Streamlit ベースの疑似トレード端末です。

## できること

- 画面を縦3分割したトレード端末レイアウト
- 左: タイムアンドセールス
- 中央上: フル板
- 中央下: 注文画面
- 右上: VWAP と保有状況
- 右中央: 選択銘柄の 1 分足
- 右下: 日経平均先物の 1 分足
- 最上部のタブで銘柄切替
- 板は 1 秒ごとに更新
- タイムアンドセールスと 1 分足は同じシミュレーションから生成
- 日経平均先物は事前設定の動き、各銘柄は日経平均と一定の相関を持つ

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud で公開する手順

1. このフォルダの中身をそのまま GitHub リポジトリへアップロードします。
2. Streamlit Community Cloud でその GitHub リポジトリを選びます。
3. Main file path を `app.py` に設定します。
4. Deploy を押せば公開できます。

## 補足

- 表示される価格・出来高・板はすべて学習用のシミュレーションです。
- 実際の市場データは使っていません。
