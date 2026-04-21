experiments/
├── set_yaml.py … `results/` 以下に実験用ディレクトリと YAML を生成（`python experiments/set_yaml.py --help`）
├── setting.md
└── results/
    ├── run_commands.txt … 全実験の `uv run python run.py --cfg_dir ...` を 1 行ずつ（`set_yaml.py` が生成）
    └── <例: deimv2_cpu_Atto_1080x1920>/
        ├── time_counter.json … 実行後に `timer.yaml` の save_dir 直下へ保存
        ├── summary.txt … 同上（`TimeCounter.save()` が平均レポートを書く）
        └── conf/ … `run.py --cfg_dir=.../conf` 向け（同一階層に processor 系 YAML）
            ├── timer.yaml
            ├── reader.yaml
            ├── displayer.yaml
            ├── detector.yaml
            ├── tracker.yaml
            └── module/
                ├── deimv2.yaml … `detector.yaml` の cfg_path（model_size / device を実験値で上書き）
                └── bytetrack.yaml … `tracker.yaml` の cfg_path（パイプライン既定のコピー）
