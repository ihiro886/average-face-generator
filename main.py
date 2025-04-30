import cv2
import dlib
import numpy as np
import sys
import os

# dlib の顔検出器とランドマーク検出器の読み込み
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

def get_landmarks(img):
    """
    画像から顔のランドマークを検出し、(68,2)のnumpy配列で返す関数
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("顔が検出できませんでした")
    # 最初に検出された顔のランドマークを使用
    shape = predictor(gray, faces[0])
    # 各パーツの座標を numpy 配列に変換
    landmarks = np.array([(pt.x, pt.y) for pt in shape.parts()], dtype=np.float32)
    return landmarks

# コマンドライン引数からファイル名と出力フォルダを取得
if len(sys.argv) != 4:
    print("Usage: python main.py <input_image1> <input_image2> <output_folder>")
    sys.exit(1)

input_img1_path = sys.argv[1]
input_img2_path = sys.argv[2]
output_folder = sys.argv[3]

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 画像の読み込み
img1 = cv2.imread(input_img1_path)
img2 = cv2.imread(input_img2_path)

if img1 is None:
    print(f"Error: Could not open image {input_img1_path}")
    sys.exit(1)
if img2 is None:
    print(f"Error: Could not open image {input_img2_path}")
    sys.exit(1)

# 画像サイズの統一（ここでは img1 のサイズに合わせる）
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 各画像のランドマークを取得
landmarks1 = get_landmarks(img1)
landmarks2 = get_landmarks(img2)

# 両画像のランドマークの平均位置を計算
avg_landmarks = (landmarks1 + landmarks2) / 2

# cv2.estimateAffinePartial2D を用いて、各画像のランドマークから平均ランドマークへのアフィン変換行列を推定
M1, _ = cv2.estimateAffinePartial2D(landmarks1, avg_landmarks)
M2, _ = cv2.estimateAffinePartial2D(landmarks2, avg_landmarks)

# 各画像を平均の形状に合わせてワーピング（リサイズ）
size = (img1.shape[1], img1.shape[0])
warped1 = cv2.warpAffine(img1, M1, size)
warped2 = cv2.warpAffine(img2, M2, size)

# ワーピングした画像のピクセルごとの平均を計算
average_face = cv2.addWeighted(warped1, 0.5, warped2, 0.5, 0)

# 出力ファイル名を生成 (入力ファイル名をハイフンで結合)
img1_filename = os.path.basename(input_img1_path)
img2_filename = os.path.basename(input_img2_path)

# 拡張子を除去
name1, ext1 = os.path.splitext(img1_filename)
name2, ext2 = os.path.splitext(img2_filename)
output_filename = f"{name1}-{name2}{ext1}" # 最初の画像の拡張子を使用

# 出力ファイルのフルパスを生成
output_img_path = os.path.join(output_folder, output_filename)

# 結果の保存
cv2.imwrite(output_img_path, average_face)

print(f"Average face image saved to {output_img_path}")