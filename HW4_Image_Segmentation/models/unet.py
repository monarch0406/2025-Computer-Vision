import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters):
    """
    A convolutional block consisting of two Conv2D layers with ReLU activation.
    """
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def encoder_block(x, filters):
    """
    Encoder block: conv_block followed by max pooling.
    Returns the skip connection and pooled output.
    """
    c = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p


def decoder_block(x, skip, filters):
    """
    Decoder block: upsample, concatenate with skip connection, then conv_block.
    """
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(input_shape=(256, 256, 3), num_classes=5):
    """
    Builds a U-Net model for semantic segmentation.

    Args:
        input_shape (tuple): Shape of the input image (H, W, C).
        num_classes (int): Number of segmentation classes.

    Returns:
        tf.keras.Model: Compiled U-Net model.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    # Bridge
    b = conv_block(p4, 1024)

    # Decoder
    d4 = decoder_block(b, c4, 512)
    d3 = decoder_block(d4, c3, 256)
    d2 = decoder_block(d3, c2, 128)
    d1 = decoder_block(d2, c1, 64)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(d1)

    model = Model(inputs, outputs, name="U-Net")
    return model
