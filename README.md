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

## 使い方

### 基本的な合成

```shell
python main.py <入力画像1のパス> <入力画像2のパス> <出力フォルダパス>
```

### 高度な合成（オプション付き）

```shell
python main.py <入力画像1のパス> <入力画像2のパス> <出力フォルダパス> --sequence --steps 10 --visualize --enhance-features
```

### オプション

- `--sequence`: モーフィングシーケンスを生成
- `--steps N`: シーケンスのステップ数（デフォルト: 5）
- `--visualize`: 三角形分割を可視化
- `--enhance-features`: 特徴強調モードで合成画像を生成

## 出力例

- `{画像1のファイル名}-{画像2のファイル名}.jpg`: 通常の合成結果
- `{画像1のファイル名}-{画像2のファイル名}_enhanced_features.jpg`: 特徴強調版
- `{画像1のファイル名}-{画像2のファイル名}_sequence_XX.jpg`: モーフィングシーケンス
- `{画像1のファイル名}-{画像2のファイル名}_triangulation.jpg`: 三角形分割の可視化
