import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)
print('TensorFlow version:', tf.__version__)

# 2) Load and Understand Dataset
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
num_classes = 10
class_names = [str(i) for i in range(num_classes)]

# Visualize sample images
plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(3, 4, i+1)  # fixed: 3 rows x 4 cols
    plt.imshow(x_train_full[i], cmap='gray')
    plt.title(f'Label: {y_train_full[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3) Data Pre-processing
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train_full = np.expand_dims(x_train_full, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
)

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# 4) Build Configurable CNN Model
def build_cnn_model(input_shape=(28, 28, 1), num_classes=10, filters=(32, 64), 
                    kernel_size=3, dense_units=128, dropout_rate=0.3, 
                    learning_rate=1e-3, optimizer_name='adam'):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    for f in filters:
        model.add(layers.Conv2D(f, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    if optimizer_name.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
        
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 5) Hyper-parameter Experiments
experiment_configs = [
    {'name': 'Exp_1_adam_k3_2layers', 'filters': (32, 64), 'kernel_size': 3, 'dropout_rate': 0.30, 'learning_rate': 1e-3, 'optimizer_name': 'adam'},
    {'name': 'Exp_2_adam_k5_2layers', 'filters': (32, 64), 'kernel_size': 5, 'dropout_rate': 0.30, 'learning_rate': 1e-3, 'optimizer_name': 'adam'},
    {'name': 'Exp_3_rmsprop_k3_3layers', 'filters': (32, 64, 128), 'kernel_size': 3, 'dropout_rate': 0.40, 'learning_rate': 1e-3, 'optimizer_name': 'rmsprop'},
    {'name': 'Exp_4_sgd_k3_3layers', 'filters': (32, 64, 128), 'kernel_size': 3, 'dropout_rate': 0.30, 'learning_rate': 1e-2, 'optimizer_name': 'sgd'},
    {'name': 'Exp_5_adam_lowLR_3layers', 'filters': (32, 64, 128), 'kernel_size': 3, 'dropout_rate': 0.25, 'learning_rate': 5e-4, 'optimizer_name': 'adam'}
]

EPOCHS = 8
BATCH_SIZE = 128
results = []
histories = {}
trained_models = {}

for cfg in experiment_configs:
    print('\n' + '=' * 70)
    print('Running:', cfg['name'])
    
    model = build_cnn_model(
        filters=cfg['filters'],
        kernel_size=cfg['kernel_size'],
        dropout_rate=cfg['dropout_rate'],
        learning_rate=cfg['learning_rate'],
        optimizer_name=cfg['optimizer_name']
    )
    
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
    
    history = model.fit(
        x_train, y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )
    
    val_loss, val_acc = model.evaluate(x_val, y_val_cat, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    
    results.append({
        'Experiment': cfg['name'],
        'Optimizer': cfg['optimizer_name'],
        'Learning Rate': cfg['learning_rate'],
        'Kernel Size': cfg['kernel_size'],
        'Conv Layers': len(cfg['filters']),
        'Filters': str(cfg['filters']),
        'Dropout': cfg['dropout_rate'],
        'Val Accuracy': val_acc,
        'Test Accuracy': test_acc,
        'Val Loss': val_loss,
        'Test Loss': test_loss
    })
    
    histories[cfg['name']] = history.history
    trained_models[cfg['name']] = model
    print(f"Validation Accuracy: {val_acc:.4f} | Test Accuracy: {test_acc:.4f}")

results_df = pd.DataFrame(results).sort_values(by='Val Accuracy', ascending=False).reset_index(drop=True)

# 6) Compare Hyper-parameter Results
plt.figure(figsize=(11, 5))
sns.barplot(data=results_df, x='Experiment', y='Val Accuracy', color='steelblue')
plt.xticks(rotation=25, ha='right')
plt.ylim(0.95, 1.0)
plt.title('Validation Accuracy by Hyper-parameter Experiment')
plt.tight_layout()
plt.show()

# 7) Final Evaluation of Best Model (Confusion Matrix)
best_experiment = results_df.loc[0, 'Experiment']
best_model = trained_models[best_experiment]

y_pred_prob = best_model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix ({best_experiment})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print('Classification Report:')
print(classification_report(y_test, y_pred, digits=4))