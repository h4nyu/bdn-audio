https://athletix.run/challenges/czaMEOxQGg

Unzip dataset into `/store`

Train models from fold 0 to 3.
Score shoud be around 0.5.

```sh
# app train <fold_index> <lr> <check_interval>
app train 0 1e-3 5
```

Check train score with all models.
```sh
# app pre-submit <fold_indices>
app pre-submit 0 1 2 3
```

Inference target with all models.
Inference Outputs are saved in `/store/submit`.

```sh
# app submit <fold_indices>
app submit 0 1 2 3
```
