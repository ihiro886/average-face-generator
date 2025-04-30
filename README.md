# average-face-generator

このアプリケーションは、2つの顔画像を合成して新しい顔画像を生成するためのツールです。

dlibを使用して顔のランドマークを検出し、三角形分割とモーフィング技術を用いて自然な合成を実現します。

## 必要条件

- OpenCV (`cv2`)
- dlib
- NumPy
- SciPy
- Matplotlib

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

dlibの顔検出モデルが別途必要です：

```shell
# モデルファイルをダウンロード
$ curl -L -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
$ bunzip2 shape_predictor_68_face_landmarks.dat.bz2
$ mv shape_predictor_68_face_landmarks.dat model/
```