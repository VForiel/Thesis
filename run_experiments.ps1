# Script to run multiple training experiments with different hyperparameters

Write-Host "Starting neural calibration experiments..."

# Baseline with new dataset
Write-Host "`n=== Experiment 1: Baseline ===" -ForegroundColor Green
python src/analysis/neural_calibration.py --samples 1000 --epochs 200 --lr 0.001 --dropout 0.05 --steps 20 --note "Baseline"

# Higher learning rate
Write-Host "`n=== Experiment 2: Higher LR ===" -ForegroundColor Green
python src/analysis/neural_calibration.py --samples 1000 --epochs 200 --lr 0.01 --dropout 0.05 --steps 20 --note "LR=0.01"

# Lower learning rate
Write-Host "`n=== Experiment 3: Lower LR ===" -ForegroundColor Green
python src/analysis/neural_calibration.py --samples 1000 --epochs 200 --lr 0.0001 --dropout 0.05 --steps 20 --note "LR=0.0001"

# Higher dropout
Write-Host "`n=== Experiment 4: Higher dropout ===" -ForegroundColor Green
python src/analysis/neural_calibration.py --samples 1000 --epochs 200 --lr 0.001 --dropout 0.2 --steps 20 --note "Dropout=0.2"

# No dropout
Write-Host "`n=== Experiment 5: No dropout ===" -ForegroundColor Green
python src/analysis/neural_calibration.py --samples 1000 --epochs 200 --lr 0.001 --dropout 0.0 --steps 20 --note "No dropout"

# More epochs
Write-Host "`n=== Experiment 6: More epochs ===" -ForegroundColor Green
python src/analysis/neural_calibration.py --samples 1000 --epochs 500 --lr 0.001 --dropout 0.05 --steps 20 --note "500 epochs"

Write-Host "`n=== All experiments complete! ===" -ForegroundColor Cyan
Write-Host "Check optimization_log.md for results"
