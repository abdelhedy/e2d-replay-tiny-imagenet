# Results

Split Tiny-ImageNet: 20 classes, 5 experiences of 4 classes, seed 42. Accuracy is the final average top-1 accuracy over all 20 classes after the last experience; forgetting is Avalanche's `StreamForgetting`.

| Strategy | Memory | Accuracy | Forgetting |
|---|---|---|---|
| Random Replay | 200 real images | 71.00 % | 24.00 % |
| E2DReplay, 1 IPC | 20 synthetic images | 44.10 % | 58.25 % |
| E2DReplay, 10 IPC | 200 synthetic images | 61.00 % | 33.88 % |
| E2DReplay, 50 IPC | 1,000 synthetic images | 69.40 % | 15.75 % |

## Per-experience accuracy, E2DReplay 10 IPC

Rows: after training on experience *i*. Columns: test accuracy (%) on the classes of experience *j*.

| After | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Mean (seen) |
|---|---|---|---|---|---|---|
| Exp 0 | 86.5 | | | | | 86.5 |
| Exp 1 | 24.0 | 96.0 | | | | 60.0 |
| Exp 2 | 25.5 | 75.0 | 86.0 | | | 62.2 |
| Exp 3 | 31.0 | 61.5 | 52.5 | 91.5 | | 59.1 |
| Exp 4 | 43.5 | 61.5 | 44.5 | 75.0 | 80.5 | **61.0** |

- The largest drop happens on Exp 0 once the buffer holds only 40 synthetic images against 2,000 real ones. Accuracy on Exp 0 partly recovers as more synthetic classes are added.
- On their own synthetic images, the teachers reach 100 % top-1 (80 % for one class).
- Total run time on 2× Tesla T4: 0.77 h. E2D recovery accounts for about 5.7 min per experience.

## Logs

Raw logs for each run go in `results/logs/`.
