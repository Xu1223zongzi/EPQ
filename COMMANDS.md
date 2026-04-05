# EPQ Command Cheat Sheet

## 0. Enter Project Directory

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
```

## 1. Single Algorithm Runs

### KCF

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF/KCF.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

### CSRT

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "CSRT/CSRT.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

### TLD

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "TLD/TLD.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

### Fusion

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "Fusion/Fusion.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

### KCF+TLD Relocalization

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF_TLD/KCF_TLD.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

## 2. Sequences With Explicit Annotation File

### bird1

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF_TLD/KCF_TLD.py" --sequence-dir "D:/论文/UAV123/data_seq/UAV123/bird1" --annotation-file "D:/论文/UAV123/anno/UAV123/bird1_1.txt" --save-video
```

### uav1

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF/KCF.py" --sequence-dir "D:/论文/UAV123/data_seq/UAV123/uav1" --annotation-file "D:/论文/UAV123/anno/UAV123/uav1_1.txt" --save-video
```

## 3. Batch Benchmark

### Small Smoke Test

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF KCF_TLD --sequence-names bike1 --max-frames 50
```

### Mid-Scale Paper Evaluation

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD FUSION KCF_TLD --sequence-names bike1 bike2 car1_s uav3 person10
```

### Full Evaluation

```powershell
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD FUSION KCF_TLD
```

## 4. Output Files

Every benchmark run creates a new folder under `benchmark_runs/benchmark_YYYYMMDD_HHMMSS/` with:

- `per_sequence_results.csv`
- `aggregate_results.csv`
- `leaderboard.md`
- `leaderboard.png`
- `success_plot.png`
- `precision_plot.png`
- `summary.json`