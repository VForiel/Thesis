# Script to run multiple training experiments with different hyperparameters
$ErrorActionPreference = 'Stop'

# Activate virtual environment
. .venv\Scripts\Activate.ps1

Write-Host "Starting neural calibration experiments..."

# Control Experiment (Gamma=10, 2000 samples, new gen)
# Write-Host "`n=== Experiment Control: Gamma=10, 2k samples ===" -ForegroundColor Green
# .venv\Scripts\python src/analysis/neural_calibration.py --samples 2000 --epochs 50 --lr 0.001 --dropout 0.05 --steps 20 --gamma 10 --batch-size 32 --note "Control Gamma 10" --no-plot

# Experiments for 100k Dataset
# We use the existing 100k dataset, loading it via subsampling logic if needed (but here we ask for 100000 directly)

# Experiment 1: Baseline 100k (BS=128)
# Write-Host "`n=== Experiment 1: 100k Samples, BS=128, LR=0.001 ===" -ForegroundColor Green
# .venv\Scripts\python src/analysis/neural_calibration.py --samples 100000 --epochs 100 --lr 0.001 --dropout 0.05 --steps 20 --gamma 9 --batch-size 128 --note "100k, BS=128" --no-plot

# Experiment 2: Larger Batch (BS=512)
# Write-Host "`n=== Experiment 2: 100k Samples, BS=512, LR=0.002 ===" -ForegroundColor Green
# .venv\Scripts\python src/analysis/neural_calibration.py --samples 100000 --epochs 100 --lr 0.002 --dropout 0.05 --steps 20 --gamma 9 --batch-size 512 --note "100k, BS=512, LR=0.002" --no-plot

# Experiment 3: Aggressive LR (BS=2048, LR=0.01) - RESCUE MODE
# Write-Host "`n=== Experiment 3: Aggressive Rescue (100k, BS=2048, LR=0.01) ===" -ForegroundColor Green
# .venv\Scripts\python src/analysis/neural_calibration.py --samples 100000 --epochs 200 --lr 0.01 --dropout 0.05 --steps 20 --gamma 9 --batch-size 2048 --note "100k, BS=2048, LR=0.01" --no-plot

# Experiment 5: Conservative Large Batch (BS=2048, LR=0.001)
# Trying to replicate success of 10k run (LR=0.001) but scaled to 100k
# Write-Host "`n=== Experiment 5: Conservative Large Batch (100k, BS=2048, LR=0.001) ===" -ForegroundColor Green
# .venv\Scripts\python src/analysis/neural_calibration.py --samples 100000 --epochs 200 --lr 0.001 --dropout 0.05 --steps 20 --gamma 9 --batch-size 2048 --note "100k, BS=2048, LR=0.001" --no-plot

# Experiment 6: High Frequency Updates (100k, BS=128, LR=0.001)
# Previous successful 10k run had 78k updates. Exp 5 had only 9k.
# Scaling updates: 100k samples / 128 BS = 781 steps/epoch. 100 epochs = 78k updates.
Write-Host "`n=== Experiment 6: High Frequency Updates (100k, BS=128, LR=0.001) ===" -ForegroundColor Green
.venv\Scripts\python src/analysis/neural_calibration.py --samples 100000 --epochs 100 --lr 0.001 --dropout 0.05 --steps 20 --gamma 9 --batch-size 128 --note "100k, BS=128, HighFreq" --no-plot

# Overfit Test (100 samples)
Write-Host "`n=== Experiment: Overfit Test (100 samples) ===" -ForegroundColor Green
.venv\Scripts\python src/analysis/neural_calibration.py --samples 100 --epochs 500 --lr 0.001 --dropout 0.0 --steps 20 --gamma 9 --batch-size 32 --note "Overfit Test" --no-plot

# Large Batch Speed Test
# Write-Host "`n=== Experiment: BS=1024, LR=0.005 ===" -ForegroundColor Green
# .venv\Scripts\python src/analysis/neural_calibration.py --samples 10000 --epochs 200 --lr 0.005 --dropout 0.05 --steps 20 --gamma 9 --batch-size 1024 --note "BS=1024" --no-plot

# Baseline 10k samples
Write-Host "`n=== Experiment 1: Baseline 10k (BS=32) ===" -ForegroundColor Green
.venv\Scripts\python src/analysis/neural_calibration.py --samples 10000 --epochs 200 --lr 0.001 --dropout 0.05 --steps 20 --gamma 9 --batch-size 32 --note "Baseline 10k" --no-plot

# Larger Batch Size
Write-Host "`n=== Experiment 2: Batch Size 128 ===" -ForegroundColor Green
.venv\Scripts\python src/analysis/neural_calibration.py --samples 10000 --epochs 200 --lr 0.001 --dropout 0.05 --steps 20 --gamma 9 --batch-size 128 --note "BS=128" --no-plot

# Higher LR with Large Batch
Write-Host "`n=== Experiment 3: BS=128, LR=0.005 ===" -ForegroundColor Green
.venv\Scripts\python src/analysis/neural_calibration.py --samples 10000 --epochs 200 --lr 0.005 --dropout 0.05 --steps 20 --gamma 9 --batch-size 128 --note "BS=128, LR=0.005" --no-plot

# Longer training
Write-Host "`n=== Experiment 4: Longer training (500 eps) ===" -ForegroundColor Green
.venv\Scripts\python src/analysis/neural_calibration.py --samples 10000 --epochs 500 --lr 0.001 --dropout 0.05 --steps 20 --gamma 9 --batch-size 64 --note "500 epochs, BS=64" --no-plot

Write-Host "`n=== All experiments complete! ===" -ForegroundColor Cyan
Write-Host "Check optimization_log.md for results"
