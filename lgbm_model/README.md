## Подготовка окружения
```
bash install_reqs.sh
```
## Адаптирование к нашему кейсу

Методы **load_train_val** из **optuna_tuner.py** и **load_test** из **inference_lgbm.py** необходимо заменить на те, которые будут выдавать данные соревнования вместо dummy данных.


## Тренировка и инференс модели
```
bash train_lgbm.sh
```

## Инференс модели
```
python3 inference_lgbm.py --exp-root ./experiments/ \
                          --exp-name path/to/exp/
```

