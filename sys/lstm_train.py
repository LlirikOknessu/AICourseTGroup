import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib


def parser_args():
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--input_dir', '-id', type=str, required=True, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, required=True, help='path to save model and results')
    parser.add_argument('--model_name', '-mn', type=str, required=True, help='name of the model to be saved')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False, help='file with dvc stage params')
    return parser.parse_args()


def build_lstm_model(input_shape, params):
    model = Sequential()
    model.add(LSTM(params['lstm_units'], input_shape=input_shape, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    return model


if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['lstm']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = args.model_name

    X_train = pd.read_csv(input_dir / 'X_train.csv')
    y_train = pd.read_csv(input_dir / 'y_train.csv')
    X_val = pd.read_csv(input_dir / 'X_val.csv')
    y_val = pd.read_csv(input_dir / 'y_val.csv')

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Reshape data to fit LSTM input
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))

    model = build_lstm_model((X_train_scaled.shape[1], X_train_scaled.shape[2]), params)
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
              epochs=params['epochs'], batch_size=params['batch_size'],
              callbacks=[early_stopping])

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / f'{model_name}.h5')
    joblib.dump(scaler, output_dir / f'{model_name}_scaler.pkl')
