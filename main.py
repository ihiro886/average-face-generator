import cv2
import dlib
import numpy as np
import sys
import os
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import argparse

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

def create_face_mask(img, landmarks):
    """顔領域のマスクを作成する関数"""
    mask = np.zeros_like(img)
    
    # 顔の輪郭点（jawライン）
    jaw_points = landmarks[0:17]
    # 眉毛から額にかけてのライン
    forehead_points = landmarks[17:27]
    
    # 顔領域の点をすべて連結
    face_points = np.vstack([jaw_points, np.flipud(forehead_points)])
    
    # 顔領域を白で塗りつぶす
    cv2.fillConvexPoly(mask, np.int32(face_points), (255, 255, 255))
    
    # マスクをグレースケールに変換してガウシアンブラーをかける（境界を滑らかに）
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_blurred = cv2.GaussianBlur(mask_gray, (31, 31), 11)
    
    # 3チャンネルに戻して正規化
    mask_normalized = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0
    
    return mask_normalized

def create_feature_enhanced_blend(img1, img2, landmarks1, landmarks2, output_folder, base_filename):
    """特徴を強調した合成画像を作成する関数"""
    # 顔のパーツごとのマスクを作成
    h, w = img1.shape[:2]
    
    # 目領域のマスク
    eye_mask = np.zeros((h, w, 3), dtype=np.float32)
    
    # 左目（ランドマーク 36-41）
    left_eye = landmarks1[36:42]
    cv2.fillConvexPoly(eye_mask, np.int32(left_eye), (1.0, 1.0, 1.0))
    
    # 右目（ランドマーク 42-47）
    right_eye = landmarks1[42:48]
    cv2.fillConvexPoly(eye_mask, np.int32(right_eye), (1.0, 1.0, 1.0))
    
    # マスクを滑らかにする
    eye_mask_gray = cv2.cvtColor(eye_mask, cv2.COLOR_BGR2GRAY)
    eye_mask = cv2.GaussianBlur(eye_mask_gray, (15, 15), 5)
    eye_mask = cv2.cvtColor(eye_mask, cv2.COLOR_GRAY2BGR)
    
    # 口領域のマスク
    mouth_mask = np.zeros((h, w, 3), dtype=np.float32)
    mouth = landmarks1[48:68]  # 口のランドマーク
    cv2.fillConvexPoly(mouth_mask, np.int32(mouth), (1.0, 1.0, 1.0))
    
    # マスクを滑らかにする
    mouth_mask_gray = cv2.cvtColor(mouth_mask, cv2.COLOR_BGR2GRAY)
    mouth_mask = cv2.GaussianBlur(mouth_mask_gray, (15, 15), 5)
    mouth_mask = cv2.cvtColor(mouth_mask, cv2.COLOR_GRAY2BGR)
    
    # 特徴強調版の合成（画像1の目、画像2の口を強調）
    eyes_enhanced = cv2.addWeighted(img1, 0.7, img2, 0.3, 0) * eye_mask + \
                    cv2.addWeighted(img1, 0.5, img2, 0.5, 0) * (1 - eye_mask)
    
    eyes_mouth_enhanced = eyes_enhanced * (1 - mouth_mask) + \
                         cv2.addWeighted(img1, 0.3, img2, 0.7, 0) * mouth_mask
    
    # 結果を保存
    output_path = os.path.join(output_folder, f"{base_filename}_enhanced_features.jpg")
    cv2.imwrite(output_path, np.uint8(eyes_mouth_enhanced))
    print(f"特徴強調版を保存しました: {output_path}")
    
    return eyes_mouth_enhanced

def create_sequence(img1, img2, landmarks1_with_boundary, landmarks2_with_boundary, tri, 
                   output_folder, base_filename, steps=5):
    """モーフィングシーケンスを生成する関数"""
    for i in range(steps + 1):
        alpha = i / float(steps)
        avg_landmarks = (1 - alpha) * landmarks1_with_boundary + alpha * landmarks2_with_boundary
        
        # 出力画像を初期化
        morphed_img = np.zeros_like(img1, dtype=np.float32)
        
        # 各三角形に対して処理
        for j in range(len(tri)):
            idx1, idx2, idx3 = tri[j]
            
            t1 = [landmarks1_with_boundary[idx1], landmarks1_with_boundary[idx2], landmarks1_with_boundary[idx3]]
            t2 = [landmarks2_with_boundary[idx1], landmarks2_with_boundary[idx2], landmarks2_with_boundary[idx3]]
            t = [avg_landmarks[idx1], avg_landmarks[idx2], avg_landmarks[idx3]]
            
            morph_triangle(img1, img2, morphed_img, t1, t2, t, alpha)
        
        # float32からuint8に変換
        morphed_img_uint8 = np.uint8(morphed_img)
        
        # 出力ファイル名
        output_path = os.path.join(output_folder, f"{base_filename}_sequence_{i:02d}.jpg")
        cv2.imwrite(output_path, morphed_img_uint8)
    
    print(f"モーフィングシーケンス ({steps+1}枚) を保存しました")

def visualize_triangulation(img, landmarks, tri, output_path):
    """三角形分割の可視化を行う関数"""
    img_copy = img.copy()
    
    # 三角形を描画
    for t in tri:
        pt1 = tuple(map(int, landmarks[t[0]]))
        pt2 = tuple(map(int, landmarks[t[1]]))
        pt3 = tuple(map(int, landmarks[t[2]]))
        
        cv2.line(img_copy, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img_copy, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img_copy, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)
    
    # ランドマークポイントを描画
    for i, point in enumerate(landmarks):
        if i < 68:  # 元のランドマークのみ描画（境界点は除外）
            x, y = point.astype(int)
            cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
    
    cv2.imwrite(output_path, img_copy)
    print(f"三角形分割の可視化を保存しました: {output_path}")

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='高度な顔合成')
    parser.add_argument('input_img1', help='入力画像1のパス')
    parser.add_argument('input_img2', help='入力画像2のパス')
    parser.add_argument('output_folder', help='出力フォルダのパス')
    parser.add_argument('--sequence', action='store_true', help='モーフィングシーケンスを生成')
    parser.add_argument('--steps', type=int, default=5, help='シーケンスのステップ数（デフォルト: 5）')
    parser.add_argument('--visualize', action='store_true', help='三角形分割を可視化')
    # parser.add_argument('--enhance-features', action='store_true', help='特徴を強調した合成を生成')
    
    args = parser.parse_args()
    
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # 画像の読み込み
    img1 = cv2.imread(args.input_img1)
    img2 = cv2.imread(args.input_img2)
    
    if img1 is None:
        print(f"エラー: 画像ファイルを開けませんでした - {args.input_img1}")
        sys.exit(1)
    if img2 is None:
        print(f"エラー: 画像ファイルを開けませんでした - {args.input_img2}")
        sys.exit(1)
    
    # 画像サイズの統一（ここでは img1 のサイズに合わせる）
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 各画像のランドマークを取得
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2_resized)
    
    if landmarks1 is None:
        print(f"エラー: {args.input_img1} から顔を検出できませんでした。")
        sys.exit(1)
    if landmarks2 is None:
        print(f"エラー: {args.input_img2} から顔を検出できませんでした。")
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
    
    # ベースファイル名の作成
    img1_filename = os.path.basename(args.input_img1)
    img2_filename = os.path.basename(args.input_img2)
    name1, ext1 = os.path.splitext(img1_filename)
    name2, ext2 = os.path.splitext(img2_filename)
    base_filename = f"{name1}-{name2}"
    
    # 三角形分割の可視化
    if args.visualize:
        vis_path = os.path.join(args.output_folder, f"{base_filename}_triangulation.jpg")
        visualize_triangulation(img1, landmarks1_with_boundary, tri, vis_path)
    
    # 出力画像を初期化
    morphed_img = np.zeros_like(img1, dtype=np.float32)
    
    # パーツごとにブレンド率を調整
    parts_alpha = {
        "eyes": 0.5,      # 目は均等ブレンド
        "nose": 0.5,      # 鼻は均等ブレンド
        "mouth": 0.5,     # 口は均等ブレンド
        "eyebrows": 0.5,  # 眉毛は均等ブレンド
        "jaw": 0.5,       # 顎は均等ブレンド
        "other": 0.5      # その他の部分
    }
    
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
    
    # 結果の保存
    output_img_path = os.path.join(args.output_folder, f"{base_filename}.jpg")
    cv2.imwrite(output_img_path, morphed_img)
    
    print(f"合成顔画像を保存しました: {output_img_path}")
    
    # 特徴強調版の生成
    # if args.enhance_features:
    #     create_feature_enhanced_blend(img1, img2_resized, landmarks1, landmarks2, 
    #                                  args.output_folder, base_filename)
    
    # モーフィングシーケンスの生成
    if args.sequence:
        create_sequence(img1, img2_resized, landmarks1_with_boundary, landmarks2_with_boundary, 
                       tri, args.output_folder, base_filename, args.steps)

if __name__ == "__main__":
    main()