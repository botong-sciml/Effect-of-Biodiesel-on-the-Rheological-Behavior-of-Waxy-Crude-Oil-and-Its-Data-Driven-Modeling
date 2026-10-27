"""Build a shear-stress-conditioned autoencoder with an extractable decoder."""

import tensorflow as tf
from tensorflow.keras import layers


class ExtractableAutoencoder:
    """Autoencoder with a separately accessible encoder and decoder.

    The encoder compresses three operating parameters (shear rate,
    temperature, and biodiesel concentration) into a latent representation L.
    The decoder reconstructs the operating parameters conditioned on L and
    shear stress, using symmetric hidden layers and optional dropout.
    """

    def __init__(self, input_dim=3, encoding_dim=1,
                 hidden_layers=None, activation='relu', dropout_rate=0.1):
        """Initialize the architecture.

        Args:
            input_dim: Number of operating parameters (default: 3).
            encoding_dim: Latent-space dimension (default: 1).
            hidden_layers: Encoder hidden-layer sizes (default: [128, 64]).
            activation: Hidden-layer activation function.
            dropout_rate: Dropout probability after each hidden layer.
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers if hidden_layers is not None else [128, 64]
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        self.build_models()
        print('Autoencoder architecture created.')

    def build_models(self):
        """Construct the encoder, decoder, and full autoencoder."""
        # Encoder: operating parameters -> latent representation L.
        encoder_input = tf.keras.Input(shape=(self.input_dim,), name='encoder_input')
        x = encoder_input
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation=self.activation,
                             name=f'encoder_dense_{i + 1}')(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate,
                                   name=f'encoder_dropout_{i + 1}')(x)

        latent_code = layers.Dense(self.encoding_dim, activation='linear',
                                   name='latent_code')(x)

        self.encoder = tf.keras.Model(encoder_input, latent_code, name='encoder')

        # Decoder: concatenate latent representation L and the shear-stress input.
        latent_input = tf.keras.Input(shape=(self.encoding_dim,), name='latent_input')
        shear_stress_input = tf.keras.Input(shape=(1,), name='shear_stress_bias')

        combined = layers.Concatenate(name='combine_latent_bias')(
            [latent_input, shear_stress_input])

        x = combined
        # Reverse the hidden-layer sizes for a symmetric decoder.
        for i, units in enumerate(reversed(self.hidden_layers)):
            x = layers.Dense(units, activation=self.activation,
                             name=f'decoder_dense_{i + 1}')(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate,
                                   name=f'decoder_dropout_{i + 1}')(x)

        decoder_output = layers.Dense(self.input_dim, activation='linear',
                                      name='decoder_output')(x)

        self.decoder = tf.keras.Model(
            inputs=[latent_input, shear_stress_input],
            outputs=decoder_output,
            name='decoder'
        )

        # Full autoencoder: reconstruct the original operating parameters.
        autoencoder_input = tf.keras.Input(shape=(self.input_dim,), name='autoencoder_input')
        shear_stress_auto = tf.keras.Input(shape=(1,), name='shear_stress_auto')

        latent = self.encoder(autoencoder_input)
        reconstructed = self.decoder([latent, shear_stress_auto])

        self.autoencoder = tf.keras.Model(
            inputs=[autoencoder_input, shear_stress_auto],
            outputs=reconstructed,
            name='autoencoder'
        )

    def compile_model(self, learning_rate=0.001):
        """Compile the autoencoder with Adam, MSE loss, and MAE/MSE metrics."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.autoencoder.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        return self.autoencoder

    def get_encoder(self):
        """Return the encoder model."""
        return self.encoder

    def get_decoder(self):
        """Return the decoder for independent use after training."""
        return self.decoder

    def get_autoencoder(self):
        """Return the full autoencoder model."""
        return self.autoencoder

    def save_components(self, filepath):
        """Save the encoder, decoder, and autoencoder separately."""
        self.encoder.save(f'{filepath}_encoder.h5')
        self.decoder.save(f'{filepath}_decoder.h5')
        self.autoencoder.save(f'{filepath}_autoencoder.h5')
        print(f'Model components saved to: {filepath}_*.h5')


class CrossValidationTracker(tf.keras.callbacks.Callback):
    """Record training and validation metrics at the end of each epoch."""

    def __init__(self, fold_num):
        super().__init__()
        self.fold_num = fold_num
        self.history = {
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': [],
            'mse': [],
            'val_mse': []
        }

    def on_epoch_end(self, epoch, logs=None):
        """Append available Keras training metrics to the fold history."""
        if logs is not None:
            for key in self.history.keys():
                if key in logs:
                    self.history[key].append(logs[key])


def create_model_pipeline(input_dim=3, encoding_dim=1,
                          hidden_layers=None, activation='relu',
                          dropout_rate=0.1, learning_rate=0.001):
    """Build and compile an extractable autoencoder.

    Returns:
        model_builder: An ExtractableAutoencoder instance for accessing components.
        model: The compiled tf.keras.Model ready for training and evaluation.
    """
    model_builder = ExtractableAutoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout_rate=dropout_rate
    )

    model = model_builder.compile_model(learning_rate=learning_rate)
    return model_builder, model
