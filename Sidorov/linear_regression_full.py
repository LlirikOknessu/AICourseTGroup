import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from joblib import dump

LINEAR_MODELS_MAPPER = {'Ridge': Ridge,
                        'LinearRegression': LinearRegression}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '_prod.csv')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    reg = LINEAR_MODELS_MAPPER.get(args.model_name)().fit(X_full, y_full)

    y_mean = y_full.mean()
    y_pred_baseline = [y_mean] * len(y_full)  # baseline
    predicted_values = np.squeeze(reg.predict(X_full))

    print("Model determination: ", reg.score(X_full, y_full))
    print("Mean apt salary: ", y_mean)

    print("Baseline MAE: ", mean_absolute_error(y_full, y_pred_baseline))
    print("Baseline MSE: ", mean_squared_error(y_full, y_pred_baseline))
    print("Baseline max_error: ", max_error(y_full, y_pred_baseline))
    baseline_min_error = 10
    for i in range(len(y_full)):
        error = abs(y_full.rating[i] - y_pred_baseline[i])
        if error.rating < baseline_min_error:
            baseline_min_error = error.rating
    print("Baseline min_error: ", baseline_min_error)

    print("Model MAE: ", mean_absolute_error(y_full, predicted_values))
    print("Model MSE: ", mean_squared_error(y_full, predicted_values))
    print("Model max_error: ", max_error(y_full, predicted_values))
    Model_min_error = 10
    for i in range(len(y_full)):
        error = abs(y_full.rating[i] - predicted_values[i])
        if error < Model_min_error:
            Model_min_error = error
    print("Model min_error: ", Model_min_error)

    intercept = reg.intercept_.astype(float)
    coefficients = reg.coef_.astype(float)
    intercept = pd.Series(intercept, name='intercept')
    coefficients = pd.Series(coefficients[0], name='coefficients')
    print("intercept:", intercept)
    print("list of coefficients:", coefficients)
    columns = [x for x in range(len(coefficients))]
    out_model = pd.DataFrame([coefficients, intercept])
    out_model.to_csv(output_model_path, index=False)
    dump(reg, output_model_joblib_path)