import cv2
import dlib
import numpy as np

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

# 画像の読み込み（適宜パスを変更してください）
img1 = cv2.imread("input/img1.jpg")
img2 = cv2.imread("input/img2.jpg")

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

# 結果の表示と保存
# cv2.imshow("Average Face", average_face)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite("output/img1-img2.jpg", average_face)
