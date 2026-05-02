import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LeakyReLU, Reshape, Input
from tensorflow.keras.datasets import fashion_mnist

# Load Dataset
(train_x, _), (_, _) = fashion_mnist.load_data()
train_x = train_x / 255.0
train_x = train_x.reshape(-1, 28, 28, 1)

# Generator
generator = Sequential([
    Input(shape=(100,)),
    Dense(512),
    LeakyReLU(negative_slope=0.2),
    BatchNormalization(momentum=0.8),
    Dense(256),
    LeakyReLU(negative_slope=0.2),
    BatchNormalization(momentum=0.8),
    Dense(128),
    LeakyReLU(negative_slope=0.2),
    BatchNormalization(momentum=0.8),
    Dense(784, activation='tanh'),
    Reshape((28, 28, 1))
])
generator.summary()

# Discriminator
discriminator = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(256),
    LeakyReLU(0.2),
    Dropout(0.5),
    Dense(128),
    LeakyReLU(0.2),
    Dropout(0.5),
    Dense(64),
    LeakyReLU(0.2),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.summary()

# GAN Model
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')
gan.summary()

# Training
epochs = 10
batch_size = 100
noise_dim = 100

for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    for i in range(train_x.shape[0] // batch_size):
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake = generator.predict(noise, verbose=0)
        real = train_x[i * batch_size : (i + 1) * batch_size]
        
        y_real = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))
        
        discriminator.trainable = True
        discriminator.train_on_batch(real, y_real)
        discriminator.train_on_batch(fake, y_fake)
        
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        discriminator.trainable = False
        gan.train_on_batch(noise, y_real)
        
    if epoch % 10 == 0:
        samples = 10
        gen = generator.predict(np.random.normal(0, 1, (samples, noise_dim)), verbose=0)
        plt.figure(figsize=(10, 2))
        for k in range(samples):
            plt.subplot(2, 5, k+1)
            plt.imshow(gen[k].reshape(28, 28), cmap='gray')
            plt.xticks([]); plt.yticks([])
        plt.show()

print('Training complete')

# Final Output visualization
noise = np.random.normal(0, 1, (10, 100))
gen = generator.predict(noise)

fig, ax = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle('Generated Images from Noise using GANs')
idx = 0
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(gen[idx].reshape(28, 28), cmap='gray')
        ax[i, j].axis('off')
        idx += 1
plt.show()