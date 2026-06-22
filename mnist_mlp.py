import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime

# ====================================
# Load MNIST Dataset
# ====================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training Images Shape:", x_train.shape)
print("Test Images Shape:", x_test.shape)

# ====================================
# Normalize Data
# ====================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ====================================
# Build Model
# ====================================

model = Sequential([
    
    Flatten(input_shape=(28, 28)),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(10, activation='softmax')
])

# ====================================
# Model Summary
# ====================================

model.summary()

# ====================================
# Compile Model
# ====================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ====================================
# Callbacks
# ====================================

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath='models/best_mnist_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# ====================================
# Train Model
# ====================================

history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        early_stopping,
        checkpoint,
        tensorboard
    ]
)

# ====================================
# Evaluate Model
# ====================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ====================================
# Save Final Model
# ====================================

model.save("mnist_mlp.keras")

print("\nModel saved successfully!")
print("File Name: mnist_mlp.keras")