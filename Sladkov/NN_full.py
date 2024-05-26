import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import datetime
import shutil
import argparse
import yaml
from tensorflow.keras import Model
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from joblib import load

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_model', '-om', type=str, default='data/models/nn_prod_version',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['NN_params']

    input_dim = 4
    output_dim = 1  
    hidden_layers = params.get('hidden_layers')  
    activation = params.get('activation') 
    learning_rate = params.get('learning_rate')
    epochs = params.get('epochs')


    BATCH_SIZE = 512
    BUFFER_SIZE = 1024

    input_dir = Path(args.input_dir)
    logs_path = Path('./data/logs') / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = Path(args.output_model)
    if logs_path.exists():
        shutil.rmtree(logs_path)
    logs_path.mkdir(parents=True)

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'


    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)



    test_ds = tf.data.Dataset.from_tensor_slices((X_full, y_full)).batch(BATCH_SIZE)

    model = Sequential(name="ModelForTGroup")
    model.add(Input(shape=(input_dim, ), name=f"input_layer"))
    for i, units in enumerate(hidden_layers, 1):
        model.add(Dense(units, activation=activation, name=f"hidden_layer_{i}"))
    model.add(Dense(output_dim, name="output_layer"))

    model.summary()


    model.compile(optimizer=Adam(learning_rate=learning_rate), metrics=['mae', 'mse'], loss='mse')
    tensorboard_callback = TensorBoard(log_dir=logs_path, histogram_freq=1)
    checkpoint_callback = ModelCheckpoint(filepath=model_path / f'{model.name}.keras', save_weights_only=False, save_best_only=True)

    model.fit(X_full, y_full, batch_size=BATCH_SIZE, epochs=epochs, validation_split =0.2, callbacks=[tensorboard_callback, checkpoint_callback])

