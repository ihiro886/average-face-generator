# average-face-generator

このアプリケーションは、2つの顔画像を合成して新しい顔画像を生成するためのツールです。

dlibを使用して顔のランドマークを検出し、三角形分割とモーフィング技術を用いて自然な合成を実現します。

## 特徴

- **基本的な合成**: 2つの顔画像を均等にブレンド
- **三角形分割による自然な変形**: 顔のパーツごとに最適な変形を適用
- **パーツごとのブレンド率調整**: 目、鼻、口などのパーツごとに異なるブレンド率を設定可能
- **モーフィングシーケンス生成**: 2つの顔画像間のモーフィングシーケンスを生成
- **三角形分割の可視化**: デバッグ用に三角形分割を可視化
<!-- - **特徴強調モード**: 特定の顔パーツの特徴をより強調した合成 -->

## 必要条件

- OpenCV (`cv2`)
- dlib
- NumPy
- SciPy

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
(.venv) $ python main.py <入力画像1のパス> <入力画像2のパス> <出力フォルダパス>
```

### 高度な合成（オプション付き）

```shell
(.venv) $ python main.py <入力画像1のパス> <入力画像2のパス> <出力フォルダパス> --sequence --steps 10 --visualize
```

### オプション

- `--sequence`: モーフィングシーケンスを生成
- `--steps N`: シーケンスのステップ数（デフォルト: 5）
- `--visualize`: 三角形分割を可視化
<!-- - `--enhance-features`: 特徴強調モードで合成画像を生成 -->

## 出力例

- `{画像1のファイル名}-{画像2のファイル名}.jpg`: 通常の合成結果
- `{画像1のファイル名}-{画像2のファイル名}_sequence_XX.jpg`: モーフィングシーケンス
- `{画像1のファイル名}-{画像2のファイル名}_triangulation.jpg`: 三角形分割の可視化
<!-- - `{画像1のファイル名}-{画像2のファイル名}_enhanced_features.jpg`: 特徴強調版 -->

## カスタマイズ

`parts_alpha` 辞書のブレンド率を調整することで、特定のパーツの特徴をより強調することができます：

```python
parts_alpha = {
    "eyes": 0.3,      # 0.3だと画像1の目の特徴が70%、画像2が30%
    "nose": 0.5,      # 均等ブレンド
    "mouth": 0.7,     # 0.7だと画像1の口の特徴が30%、画像2が70%
    "eyebrows": 0.5,  
    "jaw": 0.5,       
    "other": 0.5      
}
```

## トーナメント自動合成

複数画像をトーナメント方式で自動合成するには、`tournament_wrapper.py` を使います。  
入力ディレクトリ内の画像（2のべき乗枚必要）を自動で合成し、最終合成画像や経過を出力します。

```sh
(.venv) $ python tournament_wrapper.py <入力画像ディレクトリ> <出力ディレクトリ>
```

例:

```sh
(.venv) $ python tournament_wrapper.py input_tournament out_tournament
```

- `out_tournament/` に各ラウンドの合成画像、最終結果（`final_result.jpg`）、ログやJSONが保存されます。
- `--main-script` オプションで `main.py` のパスを変更可能です（通常は不要）。
