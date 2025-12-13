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
| 1765290251 | Epochs=100, LR=0.001, Drop=0.0, Samples=1000 | 0.500967 | 3.280714 (Train: 3.219146) | 176.82s | New dataset (2940 features), No dropout |
| 1765290767 | Epochs=200, LR=0.001, Drop=0.05, Samples=1000 | 0.499638 | 3.258723 (Train: 3.174687) | 249.14s | Baseline |
| 1765291239 | Epochs=200, LR=0.01, Drop=0.05, Samples=1000 | 0.499750 | 3.254131 (Train: 3.175655) | 316.95s | LR=0.01 |
| 1765291540 | Epochs=200, LR=0.0001, Drop=0.05, Samples=1000 | 0.499793 | 3.273988 (Train: 3.195810) | 280.88s | LR=0.0001 |
| 1765291818 | Epochs=200, LR=0.001, Drop=0.2, Samples=1000 | 0.499653 | 3.199438 (Train: 3.181276) | 261.16s | Dropout=0.2 |
| 1765292100 | Epochs=200, LR=0.001, Drop=0.0, Samples=1000 | 0.500006 | 3.263767 (Train: 3.172779) | 271.04s | No dropout |
| 1765292998 | Epochs=500, LR=0.001, Drop=0.05, Samples=1000 | 0.499544 | 3.261926 (Train: 3.176613) | 730.79s | 500 epochs |
| 1765364921 | Epochs=0, LR=0.001, Drop=0.05, Samples=10000 | 0.000000 | 3.156476 (Train: 3.311675) | 71838.97s |  |
| 1765378934 | Epochs=200, LR=0.005, Drop=0.05, Samples=10000 | 0.500009 | 3.422704 (Train: 3.255186) | 670.33s | BS=128, LR=0.005 |
| 1765380058 | Epochs=200, LR=0.005, Drop=0.05, Samples=10000 | 0.499992 | 3.315619 (Train: 3.248531) | 402.07s | BS=1024 |
| 1765381233 | Epochs=500, LR=0.001, Drop=0.0, Samples=100 | 0.011540 | 3.488529 (Train: 0.008123) | 456.28s | Overfit Test |
| 1765381235 | Epochs=200, LR=0.005, Drop=0.05, Samples=10000 | 0.868306 | 3.385726 (Train: 3.244735) | 1170.86s | BS=128, LR=0.005 |
| 1765382248 | Epochs=500, LR=0.001, Drop=0.0, Samples=100 | 0.000794 | 3.358084 (Train: 0.000455) | 238.28s | Overfit Test |
| 1765383192 | Epochs=500, LR=0.001, Drop=0.05, Samples=10000 | 0.499945 | 3.394016 (Train: 3.244432) | 4252.76s | 500 epochs, BS=64 |
| 1765383271 | Epochs=500, LR=0.001, Drop=0.0, Samples=100 | 0.064445 | 3.292390 (Train: 0.107414) | 134.68s | Overfit Test |
| 1765385279 | Epochs=500, LR=0.001, Drop=0.0, Samples=100 | 0.005424 | 3.238929 (Train: 0.005699) | 59.27s | Overfit Test |
| 1765392847 | Epochs=200, LR=0.001, Drop=0.05, Samples=10000 | 0.868300 | 3.394031 (Train: 3.244679) | 7563.25s | Baseline 10k |
| 1765393702 | Epochs=200, LR=0.001, Drop=0.05, Samples=10000 | 0.808125 | 3.342451 (Train: 2.921342) | 846.95s | BS=128 |
| 1765394511 | Epochs=200, LR=0.005, Drop=0.05, Samples=10000 | 0.780893 | 3.348138 (Train: 2.745106) | 802.05s | BS=128, LR=0.005 |
| 1765399162 | Epochs=500, LR=0.001, Drop=0.05, Samples=10000 | 0.867996 | 3.384674 (Train: 3.243977) | 4644.67s | 500 epochs, BS=64 |
| 1765455559 | Epochs=200, LR=0.01, Drop=0.05, Samples=100000 | 0.742704 | 3.198927 (Train: 2.534458) | 4700.06s | 100k, BS=2048, LR=0.01 |
| 1765455695 | Epochs=500, LR=0.001, Drop=0.0, Samples=100 | 0.607971 | 3.240529 (Train: 2.008061) | 127.65s | Overfit Test |
| 1765460495 | Epochs=200, LR=0.001, Drop=0.05, Samples=10000 | 0.868550 | 3.252630 (Train: 3.254505) | 4796.81s | Baseline 10k |
| 1765461822 | Epochs=200, LR=0.001, Drop=0.05, Samples=10000 | 0.256979 | 3.339474 (Train: 0.315763) | 1323.41s | BS=128 |
| 1765463152 | Epochs=200, LR=0.005, Drop=0.05, Samples=10000 | 0.363817 | 3.494159 (Train: 0.627280) | 1326.00s | BS=128, LR=0.005 |
| 1765469691 | Epochs=500, LR=0.001, Drop=0.05, Samples=10000 | 0.209451 | 3.252848 (Train: 0.220723) | 6535.92s | 500 epochs, BS=64 |
| 1765536766 | Epochs=200, LR=0.002, Drop=0.05, Samples=100000 | 0.696889 | 3.209935 (Train: 2.286377) | 6593.54s | 100k, LogNorm, BS=1024, LR=0.002 |
| 1765536927 | Epochs=500, LR=0.001, Drop=0.0, Samples=100 | 0.807080 | 3.336378 (Train: 2.905451) | 142.13s | Overfit Test |
| 1765542101 | Epochs=200, LR=0.001, Drop=0.05, Samples=10000 | 0.868203 | 3.332284 (Train: 3.250032) | 5169.61s | Baseline 10k |
| 1765543545 | Epochs=200, LR=0.001, Drop=0.05, Samples=10000 | 0.318954 | 3.072991 (Train: 0.484536) | 1438.34s | BS=128 |
| 1765550612 | Epochs=200, LR=0.01, Drop=0.05, Samples=100000 | 0.816699 | 3.175153 (Train: 2.958349) | 4833.58s | 100k, BS=2048, LR=0.01 |
| 1765550757 | Epochs=500, LR=0.001, Drop=0.0, Samples=100 | 0.790875 | 3.153932 (Train: 2.800320) | 136.02s | Overfit Test |
| 1765625030 | Epochs=200, LR=0.001, Drop=0.05, Samples=100000 | 0.693451 | 3.257643 (Train: 2.245032) | 4495.26s | 100k, BS=2048, LR=0.001 |
| 1765625162 | Epochs=500, LR=0.001, Drop=0.0, Samples=100 | 0.000496 | 3.075979 (Train: 0.000169) | 117.29s | Overfit Test |
| 1765629919 | Epochs=200, LR=0.001, Drop=0.05, Samples=10000 | 0.799761 | 3.317513 (Train: 2.850499) | 4754.05s | Baseline 10k |
| 1765631312 | Epochs=200, LR=0.001, Drop=0.05, Samples=10000 | 0.285134 | 3.188285 (Train: 0.377346) | 1387.92s | BS=128 |
| 1765632672 | Epochs=200, LR=0.005, Drop=0.05, Samples=10000 | 0.291358 | 3.137645 (Train: 0.431744) | 1355.72s | BS=128, LR=0.005 |
| 1765639919 | Epochs=500, LR=0.001, Drop=0.05, Samples=10000 | 0.180337 | 3.347411 (Train: 0.174075) | 7241.66s | 500 epochs, BS=64 |
