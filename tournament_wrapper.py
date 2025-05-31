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

class TournamentFaceMorph:
    def __init__(self, input_dir, output_dir, main_script_path="main.py"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.main_script_path = Path(main_script_path)
        self.log_file = None
        self.tournament_log = []
        self.tournament_bracket = {}  # トーナメント表用データ

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
            base_name = f"match_{match_num:02d}_{img1.stem}_vs_{img2.stem}"

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

    def create_tournament_bracket_html(self):
        """トーナメント表をHTMLで生成"""
        html_path = self.output_dir / "tournament_bracket.html"

        # トーナメント構造の解析
        rounds_data = {}
        for match in self.tournament_log:
            round_num = match["round"]
            if round_num not in rounds_data:
                rounds_data[round_num] = []
            rounds_data[round_num].append(match)

        # HTMLの生成
        html_content = self._generate_bracket_html(rounds_data)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.log_message(f"トーナメント表をHTMLで生成: {html_path}")

    def _generate_bracket_html(self, rounds_data):
        """トーナメント表のHTML生成"""
        max_round = max(rounds_data.keys()) if rounds_data else 0

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>顔合成トーナメント表</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .tournament-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        .tournament-title {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 24px;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .bracket {{
            display: flex;
            justify-content: center;
            align-items: center;
            min-width: 800px;
        }}
        .round {{
            display: flex;
            flex-direction: column;
            margin: 0 20px;
            position: relative;
        }}
        .round-title {{
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
            color: #555;
            font-size: 16px;
            background: #e3f2fd;
            padding: 5px 10px;
            border-radius: 15px;
        }}
        .match {{
            margin: 15px 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            border: 2px solid #ddd;
            transition: transform 0.2s;
        }}
        .match:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .participant {{
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
        }}
        .participant:last-child {{
            border-bottom: none;
        }}
        .participant.winner {{
            background-color: #e8f5e8;
            font-weight: bold;
        }}
        .participant-name {{
            flex: 1;
            font-size: 14px;
            color: #333;
        }}
        .result {{
            background: #f0f8ff;
            padding: 8px 12px;
            text-align: center;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #ddd;
        }}
        .final-result {{
            background: linear-gradient(135deg, #ffd700, #ffed4e);
            border: 3px solid #ffa000;
            transform: scale(1.1);
        }}
        .final-result .participant {{
            background: rgba(255, 215, 0, 0.2);
            font-weight: bold;
            font-size: 16px;
        }}
        .connecting-line {{
            position: absolute;
            border-top: 2px solid #4CAF50;
            z-index: -1;
        }}
        .info-section {{
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .info-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="tournament-container">
        <h1 class="tournament-title">🏆 顔合成トーナメント表</h1>

        <div class="bracket">
"""

        # 各ラウンドを生成
        for round_num in sorted(rounds_data.keys()):
            matches = rounds_data[round_num]
            is_final = round_num == max_round

            html += f'            <div class="round">\n'
            html += f'                <div class="round-title">{"決勝" if is_final else f"第{round_num}回戦"}</div>\n'

            for match in matches:
                img1_name = Path(match["image1"]).stem
                img2_name = Path(match["image2"]).stem
                result_name = Path(match["result"]).stem

                match_class = "match final-result" if is_final else "match"

                html += f'                <div class="{match_class}">\n'
                html += f'                    <div class="participant">\n'
                html += f'                        <div class="participant-name">{img1_name}</div>\n'
                html += f'                    </div>\n'
                html += f'                    <div class="participant">\n'
                html += f'                        <div class="participant-name">{img2_name}</div>\n'
                html += f'                    </div>\n'
                html += f'                    <div class="result">→ {result_name}</div>\n'
                html += f'                </div>\n'

            html += f'            </div>\n'

        # 統計情報
        total_matches = len(self.tournament_log)
        total_rounds = max_round if rounds_data else 0
        start_images = 2 ** total_rounds if total_rounds > 0 else 0

        html += f"""
        </div>

        <div class="info-section">
            <div class="info-title">📊 トーナメント統計</div>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{start_images}</div>
                    <div class="stat-label">開始画像数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{total_rounds}</div>
                    <div class="stat-label">総回戦数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{total_matches}</div>
                    <div class="stat-label">総対戦数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">1</div>
                    <div class="stat-label">最終勝者</div>
                </div>
            </div>

            <div style="margin-top: 20px;">
                <div class="info-title">📁 ファイル情報</div>
                <p><strong>入力フォルダ:</strong> {self.input_dir}</p>
                <p><strong>出力フォルダ:</strong> {self.output_dir}</p>
                <p><strong>実行日時:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

        return html

    def create_tournament_bracket_text(self):
        """トーナメント表をテキストで生成"""
        text_path = self.output_dir / "tournament_bracket.txt"

        # トーナメント構造の解析
        rounds_data = {}
        for match in self.tournament_log:
            round_num = match["round"]
            if round_num not in rounds_data:
                rounds_data[round_num] = []
            rounds_data[round_num].append(match)

        max_round = max(rounds_data.keys()) if rounds_data else 0

        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("🏆 顔合成トーナメント表\n")
            f.write("=" * 50 + "\n\n")

            for round_num in sorted(rounds_data.keys()):
                matches = rounds_data[round_num]
                is_final = round_num == max_round

                round_title = "決勝" if is_final else f"第{round_num}回戦"
                f.write(f"【{round_title}】\n")
                f.write("-" * 30 + "\n")

                for i, match in enumerate(matches, 1):
                    img1_name = Path(match["image1"]).stem
                    img2_name = Path(match["image2"]).stem
                    result_name = Path(match["result"]).stem

                    f.write(f"対戦{i}: {img1_name}\n")
                    f.write(f"     vs {img2_name}\n")
                    f.write(f"     →  {result_name}\n")
                    f.write("\n")

                f.write("\n")

            # 統計情報
            total_matches = len(self.tournament_log)
            start_images = 2 ** max_round if max_round > 0 else 0

            f.write("📊 統計情報\n")
            f.write("-" * 20 + "\n")
            f.write(f"開始画像数: {start_images}枚\n")
            f.write(f"総回戦数: {max_round}回戦\n")
            f.write(f"総対戦数: {total_matches}試合\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")

        self.log_message(f"トーナメント表をテキストで生成: {text_path}")

    def run_tournament(self):
        """トーナメント全体を実行"""
        # 出力ディレクトリの準備
        self.setup_output_directory()

        # 入力画像の取得と検証
        image_files = self.get_image_files()
        self.validate_input(image_files)

        # main.pyの存在確認
        if not self.main_script_path.exists():
            raise FileNotFoundError(f"main.pyが見つかりません: {self.main_script_path}")

        self.log_message("トーナメント開始!")

        # トーナメント実行
        current_images = image_files
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

        # トーナメント表の生成
        self.create_tournament_bracket_html()
        self.create_tournament_bracket_text()

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
