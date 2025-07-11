#!/usr/bin/env python3
"""
顔合成トーナメントラッパースクリプト

指定されたディレクトリ内の画像をトーナメント方式で合成し、
最終的な合成画像を生成します。
"""

import sys
import argparse
import subprocess
import glob
import shutil
import json
from datetime import datetime
from pathlib import Path
import os
import jinja2

class TournamentFaceMorph:
    def __init__(self, input_dir, output_dir, main_script_path="main.py"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.main_script_path = Path(main_script_path)
        self.log_file = None
        self.tournament_log = []

        # サポートする画像拡張子
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    def setup_output_directory(self):
        """出力ディレクトリの設定"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ログファイルの初期化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"tournament_log_{timestamp}.txt"

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("顔合成トーナメント実行ログ\n")
            f.write(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"入力ディレクトリ: {self.input_dir}\n")
            f.write(f"出力ディレクトリ: {self.output_dir}\n")
            f.write("="*50 + "\n\n")

    def log_message(self, message):
        """ログメッセージの出力"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")

    def get_image_files(self):
        """入力ディレクトリから画像ファイルを取得"""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(glob.glob(str(self.input_dir / f"*{ext}")))
            image_files.extend(glob.glob(str(self.input_dir / f"*{ext.upper()}")))

        image_files.sort()  # ファイル名順にソート
        return [Path(f) for f in image_files]

    def validate_input(self, image_files):
        """入力画像の検証"""
        if not image_files:
            raise ValueError(f"入力ディレクトリに画像ファイルが見つかりません: {self.input_dir}")

        # 2^n枚であることを確認
        count = len(image_files)
        if count & (count - 1) != 0:
            raise ValueError(f"画像ファイル数が2のべき乗ではありません: {count}枚")

        self.log_message(f"入力画像数: {count}枚")
        for i, img_file in enumerate(image_files):
            self.log_message(f"  {i+1}: {img_file.name}")

        return True

    def copy_input_images(self, image_files):
        """入力画像を出力ディレクトリのinputsフォルダにコピー"""
        inputs_dir = self.output_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        copied_files = []

        for img_file in image_files:
            dest_path = inputs_dir / img_file.name
            shutil.copy(str(img_file), str(dest_path))
            copied_files.append(dest_path)
            self.log_message(f"入力画像をコピー: {img_file.name} → {dest_path}")

        return copied_files

    def run_face_morph(self, img1_path, img2_path, output_folder, base_name):
        """main.pyを実行して顔合成を行う"""
        temp_output_dir = output_folder / "temp"
        temp_output_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, str(self.main_script_path),
            str(img1_path), str(img2_path), str(temp_output_dir)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 生成された合成画像を探す
            pattern = f"{img1_path.stem}-{img2_path.stem}.jpg"
            generated_file = temp_output_dir / pattern

            if generated_file.exists():
                # 出力ファイルを適切な場所に移動
                final_output = output_folder / f"{base_name}.jpg"
                shutil.move(str(generated_file), str(final_output))

                # 一時ディレクトリをクリーンアップ
                shutil.rmtree(temp_output_dir)

                return final_output
            else:
                raise FileNotFoundError(f"合成画像が生成されませんでした: {pattern}")

        except subprocess.CalledProcessError as e:
            self.log_message("エラー: main.pyの実行に失敗しました")
            self.log_message(f"stdout: {e.stdout}")
            self.log_message(f"stderr: {e.stderr}")
            raise

    def run_tournament_round(self, images, round_num):
        """トーナメントの一回戦を実行"""
        self.log_message(f"\n--- ラウンド {round_num} (対戦数: {len(images)//2}) ---")

        next_round_images = []
        round_dir = self.output_dir / f"round_{round_num:02d}"
        round_dir.mkdir(exist_ok=True)

        for i in range(0, len(images), 2):
            img1 = images[i]
            img2 = images[i + 1]

            match_num = (i // 2) + 1
            # シンプルなファイル名に変更
            base_name = f"round{round_num:02d}_match{match_num:02d}"

            self.log_message(f"  対戦 {match_num}: {img1.name} vs {img2.name}")

            try:
                # 合成実行
                output_path = self.run_face_morph(img1, img2, round_dir, base_name)
                next_round_images.append(output_path)

                # トーナメントログに記録
                match_info = {
                    "round": round_num,
                    "match": match_num,
                    "image1": str(img1),
                    "image2": str(img2),
                    "result": str(output_path),
                    "timestamp": datetime.now().isoformat()
                }
                self.tournament_log.append(match_info)

                self.log_message(f"    結果: {output_path.name}")

            except Exception as e:
                self.log_message(f"    エラー: {str(e)}")
                raise

        return next_round_images

    def save_tournament_json(self):
        """トーナメント結果をJSONで保存"""
        json_path = self.output_dir / "tournament_results.json"

        tournament_data = {
            "tournament_info": {
                "input_directory": str(self.input_dir),
                "output_directory": str(self.output_dir),
                "start_time": self.tournament_log[0]["timestamp"] if self.tournament_log else None,
                "end_time": datetime.now().isoformat(),
                "total_rounds": max([match["round"] for match in self.tournament_log]) if self.tournament_log else 0
            },
            "matches": self.tournament_log
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tournament_data, f, ensure_ascii=False, indent=2)

        self.log_message(f"トーナメント結果をJSONで保存: {json_path}")

    def generate_html(self):
        """トーナメント表をHTMLで生成"""
        template_path = Path(__file__).parent / "tournament_template.html"
        if not template_path.exists():
            raise FileNotFoundError(f"テンプレートファイルが見つかりません: {template_path}")

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=str(template_path.parent)))
        template = env.get_template(template_path.name)

        # データ準備
        total_rounds = max([match["round"] for match in self.tournament_log]) if self.tournament_log else 0
        rounds = []
        for r in range(1, total_rounds + 1):
            round_matches = [m.copy() for m in self.tournament_log if m["round"] == r]  # コピーして変更
            for match in round_matches:
                match['img1_rel'] = os.path.relpath(match['image1'], self.output_dir)
                match['img2_rel'] = os.path.relpath(match['image2'], self.output_dir)
                match['result_rel'] = os.path.relpath(match['result'], self.output_dir)
            rounds.append(round_matches)

        final_path_rel = os.path.relpath(str(self.output_dir / "final_result.jpg"), self.output_dir)

        html_content = template.render(rounds=rounds, final_result=final_path_rel)

        html_path = self.output_dir / "tournament.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.log_message(f"トーナメント表HTMLを生成: {html_path}")

    def run_tournament(self):
        """トーナメント全体を実行"""
        # 出力ディレクトリの準備
        self.setup_output_directory()

        # 入力画像の取得と検証
        image_files = self.get_image_files()
        self.validate_input(image_files)

        # 入力画像をコピー（パス問題解消のため）
        copied_image_files = self.copy_input_images(image_files)
        current_images = copied_image_files  # 以降の処理はコピー先を使用

        # main.pyの存在確認
        if not self.main_script_path.exists():
            raise FileNotFoundError(f"main.pyが見つかりません: {self.main_script_path}")

        self.log_message("トーナメント開始!")

        # トーナメント実行
        round_num = 1

        while len(current_images) > 1:
            current_images = self.run_tournament_round(current_images, round_num)
            round_num += 1

        # 最終結果
        if current_images:
            final_result = current_images[0]
            final_path = self.output_dir / "final_result.jpg"
            shutil.copy(str(final_result), str(final_path))

            self.log_message("\n🏆 トーナメント完了!")
            self.log_message(f"最終結果: {final_path}")

        # JSON結果の保存
        self.save_tournament_json()

        # HTML生成
        self.generate_html()

        # 完了ログ
        self.log_message("\n全ての処理が完了しました。")
        self.log_message(f"結果は {self.output_dir} に保存されています。")

def main():
    parser = argparse.ArgumentParser(description='顔合成トーナメント実行スクリプト')
    parser.add_argument('input_dir', help='入力画像があるディレクトリ')
    parser.add_argument('output_dir', help='出力先ディレクトリ')
    parser.add_argument('--main-script', default='main.py',
                       help='main.pyスクリプトのパス (デフォルト: main.py)')

    args = parser.parse_args()

    try:
        tournament = TournamentFaceMorph(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            main_script_path=args.main_script
        )

        tournament.run_tournament()

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
