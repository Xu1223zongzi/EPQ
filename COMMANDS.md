# EPQ Command Cheat Sheet

## 0. Enter Project Directory

Important:

- In PowerShell, run one command at a time, or separate commands with `;`
- Do not paste the `PS C:\...>` prompt text itself
- Do not put multiple commands on one line unless you add `;` between them

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ";
```

## 1. Single Algorithm Runs

### KCF

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF/KCF.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF/KCF.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

### CSRT

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "CSRT/CSRT.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "CSRT/CSRT.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

### TLD

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "TLD/TLD.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "TLD/TLD.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

### KCF+TLD Relocalization

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF_TLD/KCF_TLD.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF_TLD/KCF_TLD.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --sequence-name "bike1" --save-video
```

## 2. Sequences With Explicit Annotation File

### bird1

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF_TLD/KCF_TLD.py" --sequence-dir "D:/论文/UAV123/data_seq/UAV123/bird1" --annotation-file "D:/论文/UAV123/anno/UAV123/bird1_1.txt" --save-video
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF_TLD/KCF_TLD.py" --sequence-dir "D:/论文/UAV123/data_seq/UAV123/bird1" --annotation-file "D:/论文/UAV123/anno/UAV123/bird1_1.txt" --save-video
```

### uav1

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF/KCF.py" --sequence-dir "D:/论文/UAV123/data_seq/UAV123/uav1" --annotation-file "D:/论文/UAV123/anno/UAV123/uav1_1.txt" --save-video
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "KCF/KCF.py" --sequence-dir "D:/论文/UAV123/data_seq/UAV123/uav1" --annotation-file "D:/论文/UAV123/anno/UAV123/uav1_1.txt" --save-video
```

## 3. Batch Benchmark

### Small Smoke Test

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF KCF_TLD --sequence-names bike1 --max-frames 50
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF KCF_TLD --sequence-names bike1 --max-frames 50
```

### Mid-Scale Paper Evaluation

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD KCF_TLD --sequence-names bike1 bike2 car1_s uav3 person10
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD KCF_TLD --sequence-names bike1 bike2 car1_s uav3 person10
```

### Unified Table And Curves

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD KCF_TLD --sequence-names bike1 bike2 car1_s uav3 person10
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD KCF_TLD --sequence-names bike1 bike2 car1_s uav3 person10
```

This generates one shared result set containing all selected algorithms in:

- `aggregate_results.csv`
- `leaderboard_metrics.csv`
- `leaderboard.md`
- `leaderboard.png`
- `overlap_curve.png`
- `center_precision_curve.png`
- `normalized_precision_curve.png`
- `overlap_metrics.png`
- `localization_metrics.png`
- `stability_metrics.png`
- `efficiency_metrics.png`

### Full Evaluation

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD KCF_TLD
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD KCF_TLD
```

### Faster Benchmark With Sampling And Timeout

Step by step:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"
& "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD KCF_TLD --frame-step 2 --frame-width 480 --frame-height 360 --progress-every 50 --sequence-timeout-seconds 300 --max-sequences 5
```

Single-line version:

```powershell
Set-Location "c:\Users\ROG\PyCharmMiscProject\gitproject\EPQ"; & "c:/Users/ROG/PyCharmMiscProject/.venv/Scripts/python.exe" "uav123_benchmark.py" --uav123-root "D:/论文/UAV123/data_seq/UAV123" --algorithms KCF CSRT TLD KCF_TLD --frame-step 2 --frame-width 480 --frame-height 360 --progress-every 50 --sequence-timeout-seconds 300 --max-sequences 5
```

## 4. Output Files

Every benchmark run creates a new folder under `benchmark_runs/benchmark_YYYYMMDD_HHMMSS/` with:

- `per_sequence_results.csv`
- `aggregate_results.csv`
- `leaderboard_metrics.csv`
- `leaderboard.md`
- `leaderboard.png`
- `overlap_curve.png`
- `center_precision_curve.png`
- `normalized_precision_curve.png`
- `overlap_metrics.png`
- `localization_metrics.png`
- `stability_metrics.png`
- `efficiency_metrics.png`
- `summary.json`