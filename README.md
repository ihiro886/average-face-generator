# average-face-generator

## 環境構築

仮想環境を作成

```shell
$ python3 -m venv .venv
```

仮想環境をアクティベート

```shell
$ source .venv/bin/activate
```

ライブラリのインストール

```shell
(.venv) $ pip install -r requirements.txt
```

## モデルのダウンロードと配置

1. http://dlib.net/files/ から `shape_predictor_68_face_landmarks.dat.bz2` をダウンロードする

2. ダウンロードしたファイルを展開して、 `model` ディレクトリに配置する
