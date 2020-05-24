https://athletix.run/challenges/czaMEOxQG 1st place code


1. Unzip dataset into `/store`

2. Train models from fold 0 to 3.
The output score shoud be around 0.5.

```sh
# app train <fold_index> <lr> <check_interval>
app train 0 1e-3 5
```

3. Check training score with all models.
```sh
# app pre-submit <fold_indices>
app pre-submit 0 1 2 3
```
4. The target can be derived by all of these models.
The predicted outputs will then be saved in `/store/submit`.

```sh
# app submit <fold_indices>
app submit 0 1 2 3
```
