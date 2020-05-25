import os
import json
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.utils import plot_model
from keras.optimizers import Adam
from model.utils.smiles_tokenizer import SmilesTokenizer
from parse_config import ParseConfig


class LSTMAutoEncoder(object):
    def __init__(self):
        self.config = ParseConfig('../config.ini')
        self.model_folder = 'train_results/%s/' % self.config.model_name
        self.images_folder = os.path.join(os.path.abspath(self.model_folder), 'images')
        self.models_folder = os.path.join(os.path.abspath(self.model_folder), 'models')

        # Model shapes
        st = SmilesTokenizer()
        self.input_shape = self.config.max_len + 2 - 1, st.table_len
        self.output_dim = st.table_len

        # Model inputs/outputs variables
        self.encoder_inputs = None
        self.neck_outputs = None
        self.decode_h = None
        self.decode_c = None
        self.decoder_inputs = None
        self.decoder_outputs = None
        self.model = None
        self.smi2latent_model = None
        self.lat2states_model = None
        self.sample_model = None

    def build_encoder(self):
        self.encoder_inputs = Input(shape=self.input_shape)

        encoder = LSTM(self.config.lstm_dim, return_state=True, unroll=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)

        states = Concatenate(axis=-1)([state_h, state_c])

        neck = Dense(self.config.latent_dim, activation="relu")
        self.neck_outputs = neck(states)

    def build_decoder(self):
        self.decode_h = Dense(self.config.lstm_dim, activation="relu")
        state_h_decoded = self.decode_h(self.neck_outputs)

        self.decode_c = Dense(self.config.lstm_dim, activation="relu")
        state_c_decoded = self.decode_c(self.neck_outputs)

        self.decoder_inputs = Input(shape=self.input_shape)

        decoder_lstm = LSTM(self.config.lstm_dim, return_sequences=True, unroll=False)
        encoder_states = [state_h_decoded, state_c_decoded]
        decoder_outputs = decoder_lstm(self.decoder_inputs, initial_state=encoder_states)

        decoder_dense = Dense(self.output_dim, activation='softmax')
        self.decoder_outputs = decoder_dense(decoder_outputs)

    def build_autoencoder_model(self):
        self.build_encoder()
        self.build_decoder()
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)

        plot_model(self.model, show_shapes=True, to_file=os.path.join(self.images_folder, 'lstm_autoencoder.png'))

        optimizer = Adam(lr=self.config.train_learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        with open(os.path.join(self.models_folder, 'lstm_autoencoder_model.json'), "w") as json_file:
            json.dump(self.model.to_json(), json_file)

    def build_smi2latent_model(self):
        self.smi2latent_model = Model(self.encoder_inputs, self.neck_outputs)
        plot_model(self.smi2latent_model, show_shapes=True, to_file=os.path.join(self.images_folder, 'smi2latent_model.png'))

        with open(os.path.join(self.models_folder, 'smi2latent_model.json'), "w") as json_file:
            json.dump(self.smi2latent_model.to_json(), json_file)

    def build_lat2states_model(self):
        latent_input = Input(shape=(self.config.latent_dim,))

        state_h_decoded_2 = self.decode_h(latent_input)
        state_c_decoded_2 = self.decode_c(latent_input)
        self.lat2states_model = Model(latent_input, [state_h_decoded_2, state_c_decoded_2])
        plot_model(self.lat2states_model, show_shapes=True, to_file=os.path.join(self.images_folder, 'lat2states_model.png'))

        with open(os.path.join(self.models_folder, 'lat2states_model.json'), "w") as json_file:
            json.dump(self.lat2states_model.to_json(), json_file)

    def build_sample_model(self):
        inf_decoder_inputs = Input(batch_shape=(1, 1, self.input_shape[1]))

        inf_decoder_lstm = LSTM(self.config.lstm_dim, return_sequences=True, unroll=False, stateful=True)
        inf_decoder_outputs = inf_decoder_lstm(inf_decoder_inputs)

        inf_decoder_dense = Dense(self.output_dim, activation='softmax')
        inf_decoder_outputs = inf_decoder_dense(inf_decoder_outputs)

        self.sample_model = Model(inf_decoder_inputs, inf_decoder_outputs)
        plot_model(self.sample_model, show_shapes=True, to_file=os.path.join(self.images_folder, 'sample_model.png'))

        with open(os.path.join(self.models_folder, 'sample_model.json'), "w") as json_file:
            json.dump(self.sample_model.to_json(), json_file)
