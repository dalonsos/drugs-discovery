import os
import sys
import pickle
import shutil
import random
from tqdm import tqdm
import silence_tensorflow.auto
import tensorflow as tf
from model.lstm_autoencoder import LSTMAutoEncoder
from model.utils.utils import *
from model.callbacks import Callbacks, CustomCallback
from parse_config import ParseConfig


class Trainer(object):
    def __init__(self, train_data):
        self.train_data = train_data
        self.config = ParseConfig('../config.ini')
        np.random.seed(self.config.seed)

        print('TF version is %s ...\n' % tf.__version__)
        print('Checking if TF was built with CUDA: %s ...\n' % tf.test.is_built_with_cuda())

        if not self.config.tf_use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if tf.test.gpu_device_name():
            print('GPU found')
            print(tf.config.list_physical_devices('GPU'), '\n')
        else:
            print("No GPU found. Running on CPU")

        self.train_folder = 'train_results/%s/' % self.config.model_name
        self.st = SmilesTokenizer()
        self.smiles = None
        self.lstm_ae = None
        self.callback_list = []

        # Create model results folder
        print("\n# Creating model folder ...")

        if not os.path.exists(os.path.abspath('train_results')):
            os.mkdir(os.path.abspath('train_results'))

        if not os.path.exists(os.path.abspath(self.train_folder)):
            os.mkdir(os.path.abspath(self.train_folder))
        else:
            print("'%s' already exists. Please remove it or edit 'model_name' in config.ini ..." % self.train_folder)
            sys.exit(1)

        for folder in ['checkpoints', 'plots', 'images', 'data', 'models']:
            if not os.path.exists(os.path.join(self.train_folder, folder)):
                os.mkdir(os.path.abspath(os.path.join(self.train_folder, folder)))

        # Copy config.ini
        shutil.copy('../config.ini', self.train_folder)

    def vectorize_data(self):
        print("\n# Loading data ...")
        with open(self.train_data) as f:
            self.smiles = [s.rstrip() for s in f]
            random.shuffle(self.smiles)
            if self.config.train_num_samples < len(self.smiles):
                self.smiles = self.smiles[:self.config.train_num_samples]
        print('Num. rows: %d' % (len(self.smiles)))

        """
        # Filter valid molecules
        self.smiles = filter_valid_mols(self.smiles)
        """

        print('\n# Tokenizing SMILES ...')
        tokenized_smiles = [self.st.tokenize(smi) for smi in tqdm(self.smiles)]
        tokenized_smiles_len = [len(tokenized_smi) for tokenized_smi in tokenized_smiles]
        print('Min/Max length: %d, %d' % (min(tokenized_smiles_len), max(tokenized_smiles_len)))

        print('\n# Padding tokenized SMILES ...')
        tokenized_pad_smiles = [self.st.pad(tokenized_smi, self.config.max_len) for tokenized_smi in tqdm(tokenized_smiles)]
        tokenized_pad_smiles_len = [len(tokenized_smi) for tokenized_smi in tqdm(tokenized_pad_smiles)]
        print('Min/Max length: %d, %d' % (min(tokenized_pad_smiles_len), max(tokenized_pad_smiles_len)))

        print("\n# One hot encoding ...")
        vectorized = np.zeros(shape=(len(self.smiles), self.config.max_len + 2, self.st.table_len), dtype=np.uint8)
        for i, tok_smi in tqdm(enumerate(tokenized_pad_smiles)):
            vectorized[i] = self.st.one_hot_encode(tok_smi)

        print("\n# Save one hot vector ...")
        with open(os.path.join(self.train_folder, 'data/one_hot_encoding.pickle'), 'wb') as f:
            pickle.dump(vectorized, f)

        del vectorized

    def train(self):
        # Build model
        self.lstm_ae = LSTMAutoEncoder()
        self.lstm_ae.build_autoencoder_model()
        self.lstm_ae.build_lat2states_model()
        self.lstm_ae.build_smi2latent_model()
        self.lstm_ae.build_sample_model()

        print("\n# Loading one hot vector ...")
        with open(os.path.join(self.train_folder, 'data/one_hot_encoding.pickle'), 'rb') as f:
            vectorized = pickle.load(f)
        x = vectorized[:, :-1, :]
        y = vectorized[:, 1:, :]
        print(x.shape, y.shape)
        print(x[0, :, :])
        print("x:", "".join([self.st.int_to_char[idx] for idx in np.argmax(x[0, :, :], axis=1)]))
        print("y:", "".join([self.st.int_to_char[idx] for idx in np.argmax(y[0, :, :], axis=1)]))

        idxs = np.random.choice([i for i in range(x.shape[0])], 10000, replace=False)
        x_sample_big = x[idxs, :, :]

        with open(os.path.join(self.train_folder, 'data/x_sample_big.pickle'), 'wb') as f:
            pickle.dump(x_sample_big, f)

        with open(os.path.join(self.train_folder, 'data/x_sample_small.pickle'), 'wb') as f:
            x_sample_small = x_sample_big.copy()
            np.random.shuffle(x_sample_small)
            pickle.dump(x_sample_small[:200], f)

        cb = Callbacks(os.path.join(self.train_folder, 'checkpoints'),
                       os.path.join(self.train_folder, 'loss.csv'))

        custom_cb = CustomCallback(self.train_folder, self.config.max_len)

        self.callback_list = [cb.model_ckp,
                              cb.h,
                              cb.rlr,
                              cb.es,
                              cb.csv_logger,
                              custom_cb]

        with open(os.path.join(self.train_folder, 'validity.csv'), 'a+') as f:
            f.write('epoch;validity;uniqueness;originality\n')

        print("\n# Start training ...")
        self.lstm_ae.model.fit([x, x], y,
                               epochs=self.config.train_epochs,
                               batch_size=self.config.train_batch_size,
                               shuffle=True,
                               callbacks=self.callback_list,
                               validation_split=self.config.train_test_rate)

        print("\n# Learning plot ...")
        plt.plot(self.callback_list[1].history["loss"], label="Loss")
        plt.plot(self.callback_list[1].history["val_loss"], label="Val_Loss")
        plt.xlabel('epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend()
        plt.savefig(os.path.join(self.train_folder, 'plots/learning_plot.png'),
                    bbox_inches='tight')

        validity_df = pd.read_csv(os.path.join(self.train_folder, 'validity.csv'), sep=';')
        validity_df.plot(x='epoch', y=['validity', 'uniqueness', 'originality'], figsize=(10, 5), grid=True)
        plt.savefig(os.path.join(self.train_folder, 'plots/validity.png'),
                    bbox_inches='tight')


if __name__ == '__main__':
    trainer = Trainer('../data/dataset_cleansed.smi')
    trainer.vectorize_data()
    trainer.train()
