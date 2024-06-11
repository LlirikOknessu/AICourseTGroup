import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import datetime
import shutil
import yaml
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Load parameters from params.yaml
with open('params.yaml', 'r') as yaml_file:
    params = yaml.safe_load(yaml_file)

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Hyperparameters
input_dim = params.get('nn_training').get('input_dim')  # Input data dimensionality
output_dim = params.get('nn_training').get('output_dim')  # Output data dimensionality
hidden_layers = params.get('nn_training').get('hidden_layers')  # Neurons in each hidden layer
activation = params.get('nn_training').get('activation')  # Activation function for hidden layers
learning_rate = params.get('nn_training').get('learning_rate')
epochs = params.get('nn_training').get('epochs')
BATCH_SIZE = params.get('nn_training').get('batch_size')
BUFFER_SIZE = params.get('nn_training').get('buffer_size')

# Paths
input_dir = Path('./data/prepared')
logs_path = Path('./data/logs') / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = Path('./data/models/nn')
if logs_path.exists():
    shutil.rmtree(logs_path)
logs_path.mkdir(parents=True)

# Load data
X_train = pd.read_csv(input_dir / 'X_train.csv')
y_train = pd.read_csv(input_dir / 'y_train.csv')
X_test = pd.read_csv(input_dir / 'X_test.csv')
y_test = pd.read_csv(input_dir / 'y_test.csv')

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# Build model
model = Sequential(name="ModelForTGroup")
model.add(Input(shape=(input_dim,), name="input_layer"))  # Use Input here
for i, units in enumerate(hidden_layers, 1):
    model.add(Dense(units, activation=activation, name=f"hidden_layer_{i}"))
model.add(Dense(output_dim, name="output_layer"))

model.summary()

# Compile model
model.compile(optimizer=Adam(learning_rate=learning_rate), metrics=['mae', 'mse'], loss='mse')

# Callbacks for TensorBoard and Model Checkpoint
tensorboard_callback = TensorBoard(log_dir=logs_path, histogram_freq=1)
checkpoint_callback = ModelCheckpoint(filepath=model_path / f'{model.name}.keras', save_weights_only=False, save_best_only=True)

# Train model
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_split=params.get('nn_training').get('validation_split'), callbacks=[tensorboard_callback, checkpoint_callback])
