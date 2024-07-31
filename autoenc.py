import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, ReLU
from tensorflow.keras.models import Model

# Define the Encoder model
def build_encoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), strides=2, padding='same')(input_img)
    x = ReLU()(x)
    x = Conv2D(32, (3, 3), strides=2, padding='same')(x)
    encoded = ReLU()(x)
    encoder = Model(input_img, encoded, name='encoder')
    return encoder

# Define the Decoder model
def build_decoder(encoded_shape):
    encoded_input = Input(shape=encoded_shape)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(encoded_input)
    x = ReLU()(x)
    decoded = Conv2DTranspose(1, (3, 3), strides=2, padding='same')(x)
    decoded = tf.nn.sigmoid(decoded)
    decoder = Model(encoded_input, decoded, name='decoder')
    return decoder

# Define the Autoencoder model
def build_autoencoder(input_shape):
    encoder = build_encoder(input_shape)
    encoded_shape = encoder.output.shape[1:]
    decoder = build_decoder(encoded_shape)
    input_img = Input(shape=input_shape)
    encoded_img = encoder(input_img)
    decoded_img = decoder(encoded_img)
    autoencoder = Model(input_img, decoded_img, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder, decoder

def load_images(image_folder, target_size=(256, 256)):
    images = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
    return np.array(images)


def main():
    #Load the images
    train_images = load_images("Dataset/validation")

    # Normalize images
    train_images = train_images.astype('float32') / 255.

    # Add channel dimension
    train_images = np.expand_dims(train_images, axis=-1)

    # Build the autoencoder
    input_shape = train_images.shape[1:]
    autoencoder, encoder, decoder = build_autoencoder(input_shape)

    # Train the autoencoder
    autoencoder.fit(train_images, train_images, epochs=20, batch_size=64, shuffle=True, validation_split=0.2)

    autoencoder.save("autoencoder.h5")

if __name__ == "__main__":
    main()

