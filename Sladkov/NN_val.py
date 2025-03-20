import argparse
from pathlib import Path
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def parser_args_for_val():
    parser = argparse.ArgumentParser(description='Paths parser for validation')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--model_dir', '-md', type=str, default='data/models/nn/',
                        required=False, help='path to the saved model')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/nn_val/',
                        required=False, help='path to save validation metrics')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_val()

    input_dir = Path(args.input_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)


    output_dir.mkdir(parents=True, exist_ok=True)


    model_path = model_dir / 'ModelForTGroup.keras'
    model = load_model(model_path)


    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)


    scaler = load(input_dir / 'scaler.pkl')


    predictions = model.predict(X_val)

    y_val = scaler.inverse_transform(y_val)
    predictions = scaler.inverse_transform(predictions)


    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions) 
   

    metrics_file = output_dir / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
