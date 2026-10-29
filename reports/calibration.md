# Calibration (target ARL0 = 500 cycles, 14 sensors, rule: any sensor alarms (MEWMA: T2 > h))

Monitored in-control cycles: real (to 0.6 of life) 9379, real_early (to 0.3) 3186, simulated 187580 (20 simulated fleets with the same engine lengths).

Default source used by params.yaml "auto": **iid_est**.

| method | shape | threshold | iid | iid_est | real_early | real | real ARL0 with the iid threshold |
|---|---|---|---|---|---|---|---|
| Shewhart | - | k | 3.796 (ARL0 499) | 3.867 (ARL0 500) | 3.945 (ARL0 455) | 5.411 (ARL0 447) | 22 |
| SPRT | delta = 1.0, beta = 0.1 | alpha | 7.50e-04 (ARL0 499) | 9.30e-06 (ARL0 500) | 5.22e-06 (ARL0 637) | 3.81e-44 (ARL0 625) | 4 |
| CUSUM | kappa = 0.5 | h | 7.631 (ARL0 500) | 12.183 (ARL0 500) | 12.401 (ARL0 398) | 98.290 (ARL0 469) | 4 |
| EWMA | lam = 0.1 | L | 3.696 (ARL0 502) | 4.462 (ARL0 500) | 4.744 (ARL0 637) | 14.252 (ARL0 469) | 4 |
| MEWMA | lam = 0.1 | h | 32.464 (ARL0 502) | 48.577 (ARL0 500) | 62.971 (ARL0 531) | 447.873 (ARL0 521) | 4 |

The value in brackets is the ARL0 measured again with that threshold (a check). real_early allows only a few alarms in total, so its check is noisy.
The last column uses the textbook (iid) threshold on the real data up to 0.6 of life: how wrong you would be without calibrating on the real signal.

## Where does the difference come from? Threshold on real data vs end of the healthy window

| healthy window ends at | cycles | Shewhart | SPRT | CUSUM | EWMA | MEWMA |
|---|---|---|---|---|---|---|
| 30% of life | 3186 | 3.96 | 7.87e-06 | 12.13 | 4.77 | 61.98 |
| 40% of life | 5252 | 4.02 | 2.51e-10 | 23.09 | 6.31 | 105.92 |
| 50% of life | 7317 | 4.19 | 7.54e-22 | 50.20 | 9.00 | 204.51 |
| 60% of life | 9379 | 5.44 | 2.60e-44 | 96.97 | 14.28 | 445.44 |
