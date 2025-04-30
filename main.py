import cv2
import dlib
import numpy as np
import sys
import os
from scipy.spatial import Delaunay

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
        # 顔が検出されなかった場合はNoneを返す
        return None
    # 最初に検出された顔のランドマークを使用
    shape = predictor(gray, faces[0])
    # 各パーツの座標を numpy 配列に変換
    landmarks = np.array([(pt.x, pt.y) for pt in shape.parts()], dtype=np.float32)
    return landmarks

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    三角形領域のアフィン変換を適用する関数
    """
    # アフィン変換行列を取得
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    
    # アフィン変換を適用
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, 
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """
    三角形単位での形状のモーフィングとアルファブレンドを行う関数
    """
    # 各三角形のバウンディングボックスを計算
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    
    # オフセットポイントを取得
    t1_rect = []
    t2_rect = []
    t_rect = []
    
    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    
    # マスクを作成（三角形領域を塗りつぶす）
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)
    
    # 最初の画像の三角形パッチを適用
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    
    size = (r[2], r[3])
    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)
    
    # アルファブレンドで2つの画像を合成
    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2
    
    # マスクを使って結果を元の画像にコピー
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask


# コマンドライン引数からファイル名と出力フォルダを取得
if len(sys.argv) != 4:
    print("使い方: python main.py <入力画像1> <入力画像2> <出力フォルダ>")
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
    print(f"エラー: 画像ファイルを開けませんでした - {input_img1_path}")
    sys.exit(1)
if img2 is None:
    print(f"エラー: 画像ファイルを開けませんでした - {input_img2_path}")
    sys.exit(1)

# 画像サイズの統一（ここでは img1 のサイズに合わせる）
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 各画像のランドマークを取得
landmarks1 = get_landmarks(img1)
landmarks2 = get_landmarks(img2_resized)

if landmarks1 is None:
    print(f"エラー: {input_img1_path} から顔を検出できませんでした。")
    sys.exit(1)
if landmarks2 is None:
    print(f"エラー: {input_img2_path} から顔を検出できませんでした。")
    sys.exit(1)

# 顔の輪郭を囲む点を追加（より良いトライアンギュレーションのため）
h, w = img1.shape[:2]
boundary_points = [
    (0, 0), (w//2, 0), (w-1, 0),
    (0, h//2), (w-1, h//2),
    (0, h-1), (w//2, h-1), (w-1, h-1)
]

# ランドマークに境界点を追加
landmarks1_with_boundary = np.vstack([landmarks1, boundary_points])
landmarks2_with_boundary = np.vstack([landmarks2, boundary_points])

# ランドマーク間の平均点を計算（モーフィング用）
alpha = 0.5  # モーフィング比率（0.5で均等ブレンド）
avg_landmarks = (1 - alpha) * landmarks1_with_boundary + alpha * landmarks2_with_boundary

# 三角形分割
tri = Delaunay(avg_landmarks).simplices

# 顔パーツ領域の定義
# 目、眉毛、鼻、唇の領域インデックスを定義
eyes_indices = list(range(36, 48))  # 両目
nose_indices = list(range(27, 36))  # 鼻
mouth_indices = list(range(48, 68))  # 口
eyebrows_indices = list(range(17, 27))  # 眉毛
jaw_indices = list(range(0, 17))  # 顎

# パーツごとにブレンド率を調整（パーツの特徴をより強調）
parts_alpha = {
    "eyes": 0.5,      # 目は均等ブレンド
    "nose": 0.5,      # 鼻は均等ブレンド
    "mouth": 0.5,     # 口は均等ブレンド
    "eyebrows": 0.5,  # 眉毛は均等ブレンド
    "jaw": 0.5,       # 顎は均等ブレンド
    "other": 0.5      # その他の部分
}

# 出力画像を初期化
morphed_img = np.zeros_like(img1, dtype=np.float32)

# 各三角形に対して処理
for i in range(len(tri)):
    # 三角形の頂点インデックス
    idx1, idx2, idx3 = tri[i]
    
    # 各画像の三角形頂点を取得
    t1 = [landmarks1_with_boundary[idx1], landmarks1_with_boundary[idx2], landmarks1_with_boundary[idx3]]
    t2 = [landmarks2_with_boundary[idx1], landmarks2_with_boundary[idx2], landmarks2_with_boundary[idx3]]
    t = [avg_landmarks[idx1], avg_landmarks[idx2], avg_landmarks[idx3]]
    
    # 三角形が属する顔パーツを判断
    # 単純化のため、三角形の頂点が特定のパーツに属していれば、その三角形はそのパーツに属すると判断
    if idx1 in eyes_indices or idx2 in eyes_indices or idx3 in eyes_indices:
        curr_alpha = parts_alpha["eyes"]
    elif idx1 in nose_indices or idx2 in nose_indices or idx3 in nose_indices:
        curr_alpha = parts_alpha["nose"]
    elif idx1 in mouth_indices or idx2 in mouth_indices or idx3 in mouth_indices:
        curr_alpha = parts_alpha["mouth"]
    elif idx1 in eyebrows_indices or idx2 in eyebrows_indices or idx3 in eyebrows_indices:
        curr_alpha = parts_alpha["eyebrows"]
    elif idx1 in jaw_indices or idx2 in jaw_indices or idx3 in jaw_indices:
        curr_alpha = parts_alpha["jaw"]
    else:
        curr_alpha = parts_alpha["other"]
    
    # 三角形のモーフィングを適用
    morph_triangle(img1, img2_resized, morphed_img, t1, t2, t, curr_alpha)

# float32からuint8に変換
morphed_img = np.uint8(morphed_img)

# 出力ファイル名を生成 (入力ファイル名をハイフンで結合)
img1_filename = os.path.basename(input_img1_path)
img2_filename = os.path.basename(input_img2_path)

# 拡張子を除去
name1, ext1 = os.path.splitext(img1_filename)
name2, ext2 = os.path.splitext(img2_filename)
# 最初の画像の拡張子を使用
output_filename = f"{name1}-{name2}{ext1}"

# 出力ファイルのフルパスを生成
output_img_path = os.path.join(output_folder, output_filename)

# 結果の保存
cv2.imwrite(output_img_path, morphed_img)

print(f"合成顔画像を保存しました: {output_img_path}")

# オプション: 各パーツの強調度を調整したバージョンも生成
def create_custom_blend(img1, img2_resized, landmarks1_with_boundary, landmarks2_with_boundary, tri, 
                        eyes_alpha=0.5, nose_alpha=0.5, mouth_alpha=0.5, eyebrows_alpha=0.5, jaw_alpha=0.5,
                        suffix="_custom"):
    """カスタムブレンド率で合成顔画像を生成する関数"""
    # パーツごとのブレンド率設定
    custom_parts_alpha = {
        "eyes": eyes_alpha,
        "nose": nose_alpha,
        "mouth": mouth_alpha,
        "eyebrows": eyebrows_alpha,
        "jaw": jaw_alpha,
        "other": 0.5
    }
    
    # 平均ランドマークの計算（通常の0.5ブレンド）
    avg_landmarks = 0.5 * landmarks1_with_boundary + 0.5 * landmarks2_with_boundary
    
    # 出力画像を初期化
    custom_img = np.zeros_like(img1, dtype=np.float32)
    
    # 各三角形に対して処理
    for i in range(len(tri)):
        idx1, idx2, idx3 = tri[i]
        
        t1 = [landmarks1_with_boundary[idx1], landmarks1_with_boundary[idx2], landmarks1_with_boundary[idx3]]
        t2 = [landmarks2_with_boundary[idx1], landmarks2_with_boundary[idx2], landmarks2_with_boundary[idx3]]
        t = [avg_landmarks[idx1], avg_landmarks[idx2], avg_landmarks[idx3]]
        
        # 三角形が属する顔パーツを判断
        if idx1 in eyes_indices or idx2 in eyes_indices or idx3 in eyes_indices:
            curr_alpha = custom_parts_alpha["eyes"]
        elif idx1 in nose_indices or idx2 in nose_indices or idx3 in nose_indices:
            curr_alpha = custom_parts_alpha["nose"]
        elif idx1 in mouth_indices or idx2 in mouth_indices or idx3 in mouth_indices:
            curr_alpha = custom_parts_alpha["mouth"]
        elif idx1 in eyebrows_indices or idx2 in eyebrows_indices or idx3 in eyebrows_indices:
            curr_alpha = custom_parts_alpha["eyebrows"]
        elif idx1 in jaw_indices or idx2 in jaw_indices or idx3 in jaw_indices:
            curr_alpha = custom_parts_alpha["jaw"]
        else:
            curr_alpha = custom_parts_alpha["other"]
        
        morph_triangle(img1, img2_resized, custom_img, t1, t2, t, curr_alpha)
    
    # float32からuint8に変換
    custom_img = np.uint8(custom_img)
    
    # 出力ファイル名
    output_filename = f"{name1}-{name2}{suffix}{ext1}"
    output_img_path = os.path.join(output_folder, output_filename)
    
    # 結果の保存
    cv2.imwrite(output_img_path, custom_img)
    print(f"合成した顔画像を保存しました: {output_img_path}")
    
    return custom_img

# # 例: 目を強調したバージョン (画像1の目をより強く反映)
# create_custom_blend(img1, img2_resized, landmarks1_with_boundary, landmarks2_with_boundary, tri,
#                     eyes_alpha=0.3, suffix="_eyes1")

# # 例: 口を強調したバージョン (画像2の口をより強く反映)
# create_custom_blend(img1, img2_resized, landmarks1_with_boundary, landmarks2_with_boundary, tri,
#                     mouth_alpha=0.7, suffix="_mouth2")