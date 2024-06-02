import optuna
from tensorflow.keras import Model
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_absolute_error
from joblib import load
from sklearn.preprocessing import StandardScaler



input_dim = 4  # Размерность входных данных
output_dim = 1  # Размерность выходных данных


input_dir = Path('./data/prepared')


X_train_name = input_dir / 'X_train.csv'
y_train_name = input_dir / 'y_train.csv'
X_test_name = input_dir / 'X_test.csv'
y_test_name = input_dir / 'y_test.csv'
X_val_name = input_dir / 'X_val.csv'
y_val_name = input_dir / 'y_val.csv'

X_train = pd.read_csv(X_train_name)
y_train = pd.read_csv(y_train_name)
X_test = pd.read_csv(X_test_name)
y_test = pd.read_csv(y_test_name)
X_val = pd.read_csv(X_val_name)
y_val = pd.read_csv(y_val_name)



def objective(trial):
    # Определение гиперпараметров, которые мы хотим оптимизировать
    n_layers = trial.suggest_int('n_layers', 3, 5)  # Количество скрытых слоев
    hidden_layers = [trial.suggest_int(f'n_units_layer_{i}', 16, 256) for i in range(n_layers)]  # Количество нейронов в каждом скрытом слое
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])  # Функция активации для скрытых слоев
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)  # Скорость обучения
    BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 128, 256)  # Скорость обучения
    BUFFER_SIZE = trial.suggest_int('BUFFER_SIZE', 128, 512)  # Скорость обучения
    epochs = trial.suggest_int('epochs', 400, 800)  # Скорость обучения
    # Создание модели с новыми гиперпараметрами
    model = Sequential(name="ModelForTGroup")
    model.add(Input(shape=(input_dim, ), name=f"input_layer"))
    for i, units in enumerate(hidden_layers, 1):
        model.add(Dense(units, activation=activation, name=f"hidden_layer_{i}"))
    model.add(Dense(output_dim, name="output_layer"))

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=learning_rate), metrics=['mae', 'mse'], loss='mse')

    # Обучение модели
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data= (X_val, y_val), verbose=0)

    # Получение значения метрики для оптимизации (например, MAE)
    mse = min(history.history['val_mse'])  # Берем минимальное значение MAE на валидационном наборе
    return mse



study = optuna.create_study(direction='minimize')

# Запуск оптимизации гиперпараметров
study.optimize(objective, n_trials=300)

# Получение лучших гиперпараметров
best_params = study.best_params
print("Best hyperparameters:", best_params)
with open('optuna.txt', 'w') as file:
    file.write(str(best_params))