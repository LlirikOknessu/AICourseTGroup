import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
from sklearn.preprocessing import StandardScaler
from joblib import load

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
    output_model_path = output_dir / (args.model_name + '.csv')
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    scaler = load(input_dir / 'scaler.pkl')
    
    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    reg = LINEAR_MODELS_MAPPER.get(args.model_name)().fit(X_train, y_train)

    y_mean = y_test.mean()
    y_std = y_test.std()
    y_pred_std= np.random.normal(loc=y_mean, scale=y_std, size=len(y_test))
    predicted_values = np.squeeze(scaler.inverse_transform(reg.predict(X_test)))


    y_pred_std = scaler.inverse_transform(np.reshape(y_pred_std, (-1, 1)))
    print(reg.score(X_test, y_test))
    y_test = scaler.inverse_transform(y_test)
    print("Mean charges: ", y_mean) 

    print("STD MAE: ", mean_absolute_error(y_test, y_pred_std))
    print("Model MAE: ", mean_absolute_error(y_test, predicted_values))

    print("STD MSE: ", mean_squared_error(y_test, y_pred_std))
    print("Model MSE: ", mean_squared_error(y_test, predicted_values))
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