import argparse
import pandas as pd
from pathlib import Path
import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import joblib


def parser_args():
    parser = argparse.ArgumentParser(description='Train BP neural network model')
    parser.add_argument('--input_dir', '-id', type=str, required=True, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, required=True, help='path to save model and results')
    parser.add_argument('--model_name', '-mn', type=str, required=True, help='name of the model to be saved')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False, help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['bp_model']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = args.model_name

    X_train = pd.read_csv(input_dir / 'X_train.csv')
    y_train = pd.read_csv(input_dir / 'y_train.csv')

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Build the BP neural network model
    model = Sequential()
    model.add(Dense(params['hidden_units'], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=params['optimizer'], loss=params['loss'])

    model.fit(X_train_scaled, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / f'{model_name}.h5')
    joblib.dump(scaler, output_dir / f'{model_name}_scaler.pkl')
