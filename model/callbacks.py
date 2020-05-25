import os
import pickle
import random
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau, EarlyStopping, CSVLogger, Callback
from model.utils.utils import *
from parse_config import ParseConfig


class CustomCallback(Callback):
    """
    Check quality of generator after each training epoch
    """

    def __init__(self, folder, max_len):
        super(CustomCallback, self).__init__()
        self.folder = folder
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs=None):
        current_checkpoint = os.listdir(os.path.join(self.folder, 'checkpoints'))[-1]

        print("Loading sample ...")
        with open(os.path.join(self.folder, 'data/x_sample_small.pickle'), 'rb') as f:
            x_test_sample = pickle.load(f)

        print("Loading models ...")
        lstm_autoencoder_model = load_model_json(os.path.join(self.folder, 'models/lstm_autoencoder_model.json'))
        lstm_autoencoder_model.load_weights(os.path.join(self.folder, 'checkpoints/' + current_checkpoint))

        smi2latent_model = load_model_json(os.path.join(self.folder, 'models/smi2latent_model.json'))
        # transfer weights
        for i in range(1, 4):
            smi2latent_model.layers[i].set_weights(lstm_autoencoder_model.layers[i].get_weights())

        lat2states_model = load_model_json(os.path.join(self.folder, 'models/lat2states_model.json'))
        # transfer weights
        for i in range(1, 3):
            lat2states_model.layers[i].set_weights(lstm_autoencoder_model.layers[i + 4].get_weights())

        sample_model = load_model_json(os.path.join(self.folder, 'models/sample_model.json'))
        # transfer weights
        for i in range(1, 3):
            sample_model.layers[i].set_weights(lstm_autoencoder_model.layers[i + 6].get_weights())

        print("Get latent representation of sample ...")
        latent = smi2latent_model.predict(x_test_sample)
        print(latent[0:2])

        print("Sample from latent space ...")
        smiles_gen = [latent2smiles(lat2states_model, sample_model, latent[i:i + 1], self.max_len)
                      for i in tqdm(range(latent.shape[0]))]

        print("Get length distribution ...")
        len_arr = np.array([len(smi) for smi in smiles_gen])

        for i in range(0, 110, 10):
            per = np.percentile(len_arr, i)
            print("  percentile %d ... PC0: %d" % (i, per))

        print("Some generated sequences ...")
        random.shuffle(smiles_gen)
        for i in range(5):
            print("  %s" % smiles_gen[i])

        val, uniq, orig = check_quality_preds(smiles_gen, x_test_sample)
        with open(os.path.join(self.folder, 'validity.csv'), 'a+') as f:
            f.write('%d;%.2f;%.2f;%.2f\n' % (epoch, val, uniq, orig))


class Callbacks(object):
    def __init__(self, checkpoints_path, log_path):
        self.config = ParseConfig('../config.ini')

        self.model_ckp = ModelCheckpoint(
            filepath=os.path.join(checkpoints_path, '%s-{epoch:02d}-{val_loss:.3f}.h5' % self.config.model_name),
            monitor=self.config.monitor,
            mode=self.config.mode,
            save_best_only=self.config.save_best_only,
            save_weights_only=self.config.save_weights_only,
            verbose=self.config.verbose)

        self.h = History()

        self.rlr = ReduceLROnPlateau(monitor=self.config.monitor,
                                     factor=0.5,
                                     patience=self.config.rlr_patience,
                                     min_lr=0.000001,
                                     verbose=self.config.verbose,
                                     min_delta=1e-3)

        self.es = EarlyStopping(monitor=self.config.monitor,
                                mode=self.config.mode,
                                verbose=self.config.verbose,
                                patience=self.config.es_patience,
                                min_delta=1e-3)

        self.csv_logger = CSVLogger(log_path, append=True, separator=';')
