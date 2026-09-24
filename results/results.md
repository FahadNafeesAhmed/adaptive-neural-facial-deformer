Evaluated on 256 seeded synthetic scans per set. Errors in mm on the face region.

| set | method | surface mm | p90 | expression mm | weight MAE | rot ° | ms/frame |
|---|---|---|---|---|---|---|---|
| clean | neural | 1.22 | 1.46 | 0.70 | 0.062 | 0.51 | 1.8 |
| clean | hybrid | 0.65 | 0.77 | 0.48 | 0.055 | 0.50 | 76.5 |
| clean | icp | 4.31 | 7.70 | 3.37 | 0.247 | 3.23 | 1298.6 |
| clean | icp (true pose given) | 0.92 | 1.27 | 0.69 | 0.091 | 0.69 | 296.4 |
| noisy | neural | 1.32 | 1.67 | 0.73 | 0.059 | 0.51 | 1.9 |
| noisy | hybrid | 0.76 | 0.94 | 0.54 | 0.063 | 0.50 | 75.8 |
| noisy | icp | 4.35 | 7.72 | 3.22 | 0.244 | 3.21 | 1298.4 |
| noisy | icp (true pose given) | 1.08 | 1.54 | 0.77 | 0.107 | 0.76 | 296.1 |

Surface error (mm) on noisy scans by head yaw:

| method | yaw 0-10° | yaw 10-20° | yaw 20-35° |
|---|---|---|---|
| neural | 1.24 | 1.33 | 1.36 |
| hybrid | 0.73 | 0.76 | 0.77 |
| icp | 3.80 | 4.14 | 4.85 |
| icp (true pose given) | 1.00 | 1.04 | 1.17 |

Network latency, single scan: 39.0 ms
