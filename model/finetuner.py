import os
import sys
import pickle
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import silence_tensorflow.auto
import tensorflow as tf
from keras.optimizers import Adam
from model.lstm_autoencoder import LSTMAutoEncoder
from model.utils.utils import *
from model.callbacks import Callbacks, CustomCallback
from parse_config import ParseConfig


class FineTuner(object):
    def __init__(self, ft_data):
        self.ft_data = ft_data
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
        self.checkpoint_name = None
        self.lstm_autoencoder_model = None

        # Create finetune results folder
        print("\n# Creating finetune folder ...")

        if not os.path.exists('train_results/%s/finetune_results' % self.config.model_name):
            os.mkdir('train_results/%s/finetune_results' % self.config.model_name)

        """"
        Get next finetune model folder name. I.e. if the last
        finetune folder is 'train_results/covid19/finetune_results/ft_01'
        then this functions will return 'ft_03'
        """

        folders = os.listdir('train_results/%s/finetune_results' % self.config.model_name)
        ft_folders = [f for f in folders if f.startswith('ft_')]
        if not ft_folders:
            self.ft_folder = 'train_results/%s/finetune_results/ft_01' % self.config.model_name
        else:
            last_ft_folder = sorted(ft_folders)[-1]
            _, num = last_ft_folder.split('_')
            num_next = str(int(num) + 1)
            if len(num_next) == 1:
                num_next = '0' + num_next
            self.ft_folder = 'train_results/%s/finetune_results/ft_%s' % (self.config.model_name, num_next)

        os.mkdir(self.ft_folder)

        for folder in ['checkpoints', 'plots', 'data', 'models']:
            if not os.path.exists(os.path.join(self.ft_folder, folder)):
                os.mkdir(os.path.abspath(os.path.join(self.ft_folder, folder)))

        # Copy all json model files from train to finetune folder
        shutil.copy(os.path.join(self.train_folder, 'models/lstm_autoencoder_model.json'),
                    os.path.join(self.ft_folder, 'models'))
        shutil.copy(os.path.join(self.train_folder, 'models/lat2states_model.json'),
                    os.path.join(self.ft_folder, 'models'))
        shutil.copy(os.path.join(self.train_folder, 'models/sample_model.json'),
                    os.path.join(self.ft_folder, 'models'))
        shutil.copy(os.path.join(self.train_folder, 'models/smi2latent_model.json'),
                    os.path.join(self.ft_folder, 'models'))

    def vectorize_data(self):
        print("\n# Loading data ...")
        with open(self.ft_data) as f:
            self.smiles = [s.rstrip() for s in f]
            if self.config.ft_num_samples < len(self.smiles):
                self.smiles = self.smiles[:self.config.ft_num_samples]
        print('Num. rows: %d' % (len(self.smiles)))

        # Filter valid molecules
        self.smiles = filter_valid_mols(self.smiles)

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
        with open(os.path.join(self.ft_folder, 'data/one_hot_encoding.pickle'), 'wb') as f:
            pickle.dump(vectorized, f)

    def finetune(self):
        # Load models
        self.checkpoint_name = get_best_model_name(self.config.model_name,
                                                   os.path.join(self.train_folder, 'loss.csv'))

        print("\n# Loading models from checkpoint '%s' ..." % self.checkpoint_name)
        self.lstm_autoencoder_model = load_model_json(
            os.path.join(self.train_folder, 'models/lstm_autoencoder_model.json'))
        self.lstm_autoencoder_model.load_weights(os.path.join(self.train_folder, 'checkpoints/' + self.checkpoint_name))

        optimizer = Adam(lr=self.config.ft_learning_rate)
        self.lstm_autoencoder_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        print("\n# Loading one hot vector ...")
        with open(os.path.join(self.ft_folder, 'data/one_hot_encoding.pickle'), 'rb') as f:
            vectorized = pickle.load(f)
        x = vectorized[:, :-1, :]
        y = vectorized[:, 1:, :]
        print(x.shape, y.shape)
        print(x[0, :, :])
        print("x:", "".join([self.st.int_to_char[idx] for idx in np.argmax(x[0, :, :], axis=1)]))
        print("y:", "".join([self.st.int_to_char[idx] for idx in np.argmax(y[0, :, :], axis=1)]))

        idxs = np.random.choice([i for i in range(x.shape[0])], x.shape[0], replace=False)
        x_sample_big = x[idxs, :, :]

        with open(os.path.join(self.ft_folder, 'data/x_sample_big.pickle'), 'wb') as f:
            pickle.dump(x_sample_big, f)

        with open(os.path.join(self.ft_folder, 'data/x_sample_small.pickle'), 'wb') as f:
            x_sample_small = x_sample_big.copy()
            np.random.shuffle(x_sample_small)
            pickle.dump(x_sample_small[:x.shape[0]], f)

        cb = Callbacks(os.path.join(self.ft_folder, 'checkpoints'),
                       os.path.join(self.ft_folder, 'loss.csv'))

        custom_cb = CustomCallback(self.ft_folder, self.config.max_len)

        self.callback_list = [cb.model_ckp,
                              cb.h,
                              cb.rlr,
                              cb.es,
                              cb.csv_logger,
                              custom_cb]

        with open(os.path.join(self.ft_folder, 'validity.csv'), 'a+') as f:
            f.write('epoch;validity;uniqueness;originality\n')

        print("\n# Start fine tuning ...")
        self.lstm_autoencoder_model.fit([x, x], y,
                                        epochs=self.config.ft_epochs,
                                        batch_size=self.config.ft_batch_size,
                                        shuffle=True,
                                        callbacks=self.callback_list,
                                        validation_split=self.config.ft_test_rate)

        print("\n# Learning plot ...")
        plt.plot(self.callback_list[1].history["loss"], label="Loss")
        plt.plot(self.callback_list[1].history["val_loss"], label="Val_Loss")
        plt.xlabel('epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend()
        plt.savefig(os.path.join(self.ft_folder, 'plots/learning_plot.png'),
                    bbox_inches='tight')

        validity_df = pd.read_csv(os.path.join(self.ft_folder, 'validity.csv'), sep=';')
        validity_df.plot(x='epoch', y=['validity', 'uniqueness', 'originality'], figsize=(10, 5), grid=True)
        plt.savefig(os.path.join(self.ft_folder, 'plots/validity.png'),
                    bbox_inches='tight')


if __name__ == '__main__':
    ft = FineTuner('../data/generations/binding_results/gen_08/finetune_08.smi')
    ft.vectorize_data()
    ft.finetune()
