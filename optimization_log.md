# Optimization Log

| Iteration | Changes | Final Loss | Test MSE | Duration | Notes |
|-----------|---------|------------|----------|----------|-------|
| 1765016840 | Epochs=500, LR=0.001, Drop=0.05, Samples=2000 | 0.057621 | 3.072321 | 2132.35s | Baseline run |
| 1765017093 | Epochs=50, LR=0.005, Drop=0.05, Samples=2000 | 0.189809 | 2.990911 | 143.34s | Faster run, higher LR |
| 1765017872 | Epochs=200, LR=0.001, Drop=0.05, Samples=100 | 0.131787 | 3.424460 | 34.23s | Small dataset baseline |
| 1765018004 | Epochs=100, LR=0.002, Drop=0.05, Samples=2000 | 0.108236 | 3.187204 | 65.42s | MLP architecture, 2000 samples |
| 1765018452 | Epochs=200, LR=0.002, Drop=0.05, Samples=2000 | 0.052876 | 3.285113 | 119.71s | MLP, MSE Loss, 2000 samples |
| 1765018580 | Epochs=100, LR=0.002, Drop=0.05, Samples=2000 | 0.066547 | 2.986523 (Train: 0.038851) | 66.37s | MLP, MSE Loss, Check Train MSE |
| 1765018760 | Epochs=200, LR=0.002, Drop=0.2, Samples=2000 | 0.175128 | 3.207685 (Train: 0.255003) | 78.08s | Smaller MLP, Dropout 0.2 |
| 1765019533 | Epochs=200, LR=0.002, Drop=0.05, Samples=2000 | 0.123650 | 3.363702 (Train: 0.169779) | 390.00s | Smaller CNN, CosineLoss, Cached Dataset |
| 1765019845 | Epochs=100, LR=0.002, Drop=0.05, Samples=2000 | 0.088922 | 3.220194 (Train: 0.040709) | 82.50s | MLP, UnitCircleMSELoss |
| 1765020176 | Epochs=100, LR=0.002, Drop=0.3, Samples=2000 | 0.185352 | 3.199596 (Train: 0.220972) | 90.70s | MLP, Tanh, Dropout 0.3 |
| 1765020290 | Epochs=100, LR=0.0005, Drop=0.05, Samples=2000 | 0.125201 | 3.138655 (Train: 0.076354) | 54.41s | Simple MLP, Low LR |
| 1765020677 | Epochs=100, LR=0.001, Drop=0.05, Samples=2000 | 0.057911 | 2.823782 (Train: 0.049313) | 35.16s | MLP, No BN |
| 1765029005 | Epochs=100, LR=0.001, Drop=0.1, Samples=2000 | 0.091302 | 3.020779 (Train: 0.090919) | 32.56s | MLP, No BN, Dropout 0.1 |
| 1765029418 | Epochs=100, LR=0.001, Drop=0.0, Samples=2000 | 0.019006 | 3.165298 (Train: 0.009269) | 252.92s | Large MLP, No BN, No Dropout |
| 1765032394 | Epochs=1000, LR=0.001, Drop=0.05, Samples=2000 | 0.026932 | 3.000971 (Train: 0.020805) | 356.74s |  |
| 1765033698 | Epochs=100, LR=0.001, Drop=0.0, Samples=2000 | 0.354276 | 3.190222 (Train: 1.075751) | 21.88s | Tiny MLP (64-32), No Drop |
| 1765033789 | Epochs=100, LR=0.001, Drop=0.0, Samples=2000 | 0.306534 | 3.228515 (Train: 1.045011) | 25.48s | Tiny MLP, MSE Loss |
