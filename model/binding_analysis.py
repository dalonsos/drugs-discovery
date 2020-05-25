import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import configparser
import pickle
from rdkit.Chem.QED import qed
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import silence_tensorflow.auto
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import History, ReduceLROnPlateau
from keras.utils import plot_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from model.utils.utils import *
from model.utils.smiles_tokenizer import SmilesTokenizer
from parse_config import ParseConfig

config = ParseConfig('../config.ini')

gen_files = os.listdir('gen_results/%s/data' % config.model_name)
gen_files = [f for f in gen_files if f.startswith('generated_')]
last_gen_file = sorted(gen_files)[-1]
GENERATION_ID = last_gen_file.split('_')[-1].split('.')[0]
GENERATIONS_PATH = '../data/generations'
BINDING_RESULTS_PATH = '../data/generations/binding_results/gen_%s' % GENERATION_ID
BINDING_RESULTS_FILE = 'binding_scores_%s.csv' % GENERATION_ID
PLOTS_PATH = os.path.join(BINDING_RESULTS_PATH, 'plots')
BINDING_SCORE_MODEL_PATH = os.path.join(BINDING_RESULTS_PATH, 'model')
MODEL_NAME = config.model_name
TRAIN_FOLDER = os.path.abspath('train_results/%s/' % MODEL_NAME)
FINETUNE_FOLDER = 'train_results/%s/finetune_results/ft_%s' % (MODEL_NAME, GENERATION_ID)
MAX_LEN = config.max_len

# Edit
NUM_GEN_SAMPLES = 2000
PC0_BOUNDS = (-40, 120)
PC1_BOUNDS = (-40, 40)
MASTER_THRESH = -9.0
GEN_THRESH = -9.5

np.random.seed(config.seed)


def plot_lr_line(series1, series2, colors, var1_name, var2_name):
    fig, ax = plt.subplots()

    coef = np.polyfit(np.array(series1), np.array(series2), 1)
    poly1d_fn = np.poly1d(coef)

    ax.scatter(np.array(series1), np.array(series2), color=colors, alpha=0.5)
    ax.plot(np.array(series1), poly1d_fn(np.array(series1)), '--k')
    ax.set_xlabel(var1_name, fontsize=12)
    ax.set_ylabel(var2_name, fontsize=12)

    plt.savefig(os.path.join(PLOTS_PATH, '%s_vs_%s.png' % (var1_name, var2_name)), bbox_inches='tight')


def load_autoencoder():
    if GENERATION_ID == '00':
        folder = TRAIN_FOLDER
    else:
        folder = FINETUNE_FOLDER

    # Load models
    checkpoint_name = get_best_model_name(config.model_name, os.path.join(folder, 'loss.csv'))

    print("\n# Loading models from checkpoint '%s' ..." % checkpoint_name)
    lstm_autoencoder_model = load_model_json(os.path.join(folder, 'models/lstm_autoencoder_model.json'))
    lstm_autoencoder_model.load_weights(os.path.join(folder, 'checkpoints/' + checkpoint_name))

    smi2latent_model = load_model_json(os.path.join(folder, 'models/smi2latent_model.json'))
    # transfer weights
    for i in range(1, 4):
        smi2latent_model.layers[i].set_weights(lstm_autoencoder_model.layers[i].get_weights())

    lat2states_model = load_model_json(os.path.join(folder, 'models/lat2states_model.json'))
    # transfer weights
    for i in range(1, 3):
        lat2states_model.layers[i].set_weights(lstm_autoencoder_model.layers[i + 4].get_weights())

    sample_model = load_model_json(os.path.join(folder, 'models/sample_model.json'))
    # transfer weights
    for i in range(1, 3):
        sample_model.layers[i].set_weights(lstm_autoencoder_model.layers[i + 6].get_weights())

    return lstm_autoencoder_model, \
           smi2latent_model, \
           lat2states_model, \
           sample_model


def train_bindscore_model(x, y):
    print("\n# Model binding score from the latent space ...")

    # Split test into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=config.seed)

    bindscore_model = Sequential()
    bindscore_model.add(Dense(128, input_shape=(16,), activation="relu"))
    bindscore_model.add(Dropout(0.1))
    bindscore_model.add(Dense(64, activation="relu"))
    bindscore_model.add(Dense(1))

    optimizer = Adam(lr=0.001)
    bindscore_model.compile(optimizer=optimizer, loss="mse")

    plot_model(bindscore_model, show_shapes=True,
               to_file=os.path.join(BINDING_SCORE_MODEL_PATH, 'binding_score_model.png'))

    with open(os.path.join(BINDING_SCORE_MODEL_PATH, 'binding_score_model.json'), "w") as json_file:
        json.dump(bindscore_model.to_json(), json_file)

    h = History()
    rlr = ReduceLROnPlateau(monitor="val_loss",
                            factor=0.5,
                            patience=10,
                            min_lr=0.000001,
                            verbose=True,
                            min_delta=1e-3)

    bindscore_model.fit(x_train,
                        y_train,
                        batch_size=64,
                        epochs=120,
                        callbacks=[h, rlr],
                        validation_data=[x_test, y_test])

    bindscore_model.save_weights(os.path.join(BINDING_SCORE_MODEL_PATH, 'binding_score_model.h5'))

    print("\n# Learning plot ...")
    fig, ax = plt.subplots()
    plt.plot(h.history["loss"], label="loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_PATH, 'learning_plot.png'), bbox_inches='tight')

    y_train_pred = bindscore_model.predict(x_train)
    y_test_pred = bindscore_model.predict(x_test)

    fig, ax = plt.subplots()
    plt.scatter(y_train, y_train_pred, label="train")
    plt.scatter(y_test, y_test_pred, label="test")
    plt.legend()
    plt.xlabel('real score')
    plt.ylabel('pred score')
    plt.savefig(os.path.join(PLOTS_PATH, 'bindscore_latent_model.png'), bbox_inches='tight')

    r_train = np.corrcoef(y_train, y_train_pred.squeeze(axis=1))[0, 1]
    r_test = np.corrcoef(y_test, y_test_pred.squeeze(axis=1))[0, 1]
    print('Correlation train / test: %.3f / %.3f' % (r_train, r_test))


def gen_low_score_mols(latent,
                       lat2states_model,
                       sample_model,
                       bindscore_model,
                       master_updated_df,
                       pc0_bounds,
                       pc1_bounds):
    print("\n# Generate low score molecules ...")

    # Get 2D latent representation
    print("  Apply PCA to get a 2D latent space ...")
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)
    print(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_))

    print("  Plot predictions scatter plot ...")
    fig, ax = plt.subplots()
    scat = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=master_updated_df['score_pred'], s=50, marker='o')
    fig.colorbar(scat)
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')
    plt.savefig(os.path.join(BINDING_RESULTS_PATH, 'plots/latent_2d_preds_scatter.png'), bbox_inches='tight')

    """
    Visually explore the scatter plot of the predicted scores with regard to the latent 2D representation.
    Get the boundaries where the lowest predicted scores are located in order to sample from that region
    """
    print("  Random uniform sample from region ...")

    # Random uniform sample
    xy_min = [pc0_bounds[0], pc1_bounds[0]]
    xy_max = [pc0_bounds[1], pc1_bounds[1]]
    latent_2d_gen = np.random.uniform(low=xy_min, high=xy_max, size=(NUM_GEN_SAMPLES, 2))

    # Convert back generated 2D latent data to its original dimension
    print("  Generate valid smiles ...")
    latent_gen = pca.inverse_transform(latent_2d_gen)

    # Generate valid smiles from latent
    smiles_gen = [latent2smiles(lat2states_model, sample_model, latent_gen[i:i + 1], config.max_len)
                  for i in tqdm(range(latent_gen.shape[0]))]

    # Get score prediction for generated molecules and filter
    print("  Binding score predictions for generated molecules ...")
    y_pred_gen = bindscore_model.predict(latent_gen)
    y_pred_gen = y_pred_gen.squeeze(axis=1)

    for i in range(10, 100, 10):
        per = np.percentile(y_pred_gen, i, axis=0)
        print("    percentile %d ... score: %.3f" % (i, per))

    gen_df = pd.DataFrame({'smile': pd.Series(smiles_gen), 'score_pred': pd.Series(y_pred_gen)})

    return gen_df


if __name__ == '__main__':
    print('TF version is %s ...\n' % tf.__version__)
    print('Checking if TF was built with CUDA: %s ...\n' % tf.test.is_built_with_cuda())

    if not config.tf_use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
        print(tf.config.list_physical_devices('GPU'), '\n')
    else:
        print("No GPU found. Running on CPU")

    if not os.path.exists(PLOTS_PATH):
        os.mkdir(PLOTS_PATH)

    # Read simulation results
    print("\n# Read simulation results and master table ...")
    print("scores table:")
    scores_df = pd.read_csv(os.path.join(BINDING_RESULTS_PATH, BINDING_RESULTS_FILE),
                            sep=',',
                            dtype={'gen': 'str'})
    print(scores_df.head(5))

    # Read master table
    master_df = pd.read_csv(os.path.join(GENERATIONS_PATH, 'master_table.csv'),
                            sep=',',
                            dtype={'gen': 'str'})
    master_df.drop_duplicates(subset=['smile'], keep='first', inplace=True)
    master_df.reset_index(inplace=True, drop=True)
    print("master table:")
    print(master_df.head(5))

    # For each molecule get the configuration with the lowest binding affinity
    print("\n# For each molecule get the configuration with the lowest binding affinity ...")
    scores_df = scores_df.groupby("Ligand").min()["Binding Affinity"].reset_index()
    scores_df['id'] = scores_df['Ligand'].str.split("_").str[2]
    scores_df['gen'] = scores_df['Ligand'].str.split("_").str[4]
    scores_df['score'] = scores_df["Binding Affinity"]
    scores_df = scores_df[['id', 'gen', 'score']]
    scores_df.drop_duplicates(subset=['id', 'gen'], keep='first', inplace=True)
    scores_df.reset_index(inplace=True, drop=True)
    print(scores_df.head(5))

    # Join master table and the simulation results by 'id' and 'gen'
    print("\n# Join simulation results and master table ...")
    master_updated_df = pd.merge(master_df, scores_df, on=['id', 'gen'], how='inner', suffixes=('_old', '_new'))
    master_updated_df['score'] = np.where(master_updated_df['score_new'].isnull(),
                                          master_updated_df['score_old'],
                                          master_updated_df['score_new'])
    master_updated_df = master_updated_df.drop(['score_old', 'score_new'], axis=1)
    master_updated_df = master_updated_df.sort_values('score', ascending=True)
    master_updated_df.reset_index(inplace=True, drop=True)

    # Before writing to csv, check if a previous master table scores file exists in order to append data
    if not os.path.exists(os.path.join(GENERATIONS_PATH, 'master_table_scores.csv')):
        master_updated_df.to_csv(os.path.join(GENERATIONS_PATH, 'master_table_scores.csv'), sep=',', index=False)
    else:
        master_updated_prev_df = pd.read_csv(os.path.join(GENERATIONS_PATH, 'master_table_scores.csv'),
                                             sep=',',
                                             dtype={'gen': 'str'})
        master_updated_df = pd.concat([master_updated_prev_df, master_updated_df])
        # Remove duplicates
        master_updated_df.drop_duplicates(keep='first', inplace=True)
        master_updated_df.reset_index(inplace=True, drop=True)
        master_updated_df.to_csv(os.path.join(GENERATIONS_PATH, 'master_table_scores.csv'), sep=',', index=False)

    # Min score by source
    score_min_by_source = master_updated_df.groupby('source')['score'].min()
    print("Min. binding score grouped by source:")
    print(score_min_by_source)

    """
    # Filter master table updated
    master_updated_df = master_updated_df[master_updated_df['source'] == 'generated']
    master_updated_df.reset_index(inplace=True, drop=True)
    print(master_updated_df.head(5))
    """

    # Distributions
    print("\n# Get distributions ...")
    fig, ax = plt.subplots()
    sns.kdeplot(master_updated_df[master_updated_df['source'] == 'candidates']['score'], label='candidates')
    sns.kdeplot(master_updated_df[master_updated_df['source'] == 'hiv']['score'], label='hiv')
    sns.kdeplot(master_updated_df[master_updated_df['source'] == 'training']['score'], label='training')
    generations = list(master_updated_df['gen'].unique())
    for gen in generations:
        sns.kdeplot(master_updated_df[(master_updated_df['source'] == 'generated') &
                                      (master_updated_df['gen'] == gen)]['score'], label='gen %s' % gen)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_PATH, 'bindscore_dist.png'), bbox_inches='tight')

    # A high correlation between molecular weight and binding score is observed.
    # The higher the molecular weight the lower the binding score

    print("\n# Linear regression of scores with respect different variables ...")
    plot_lr_line(master_updated_df['weight'],
                 master_updated_df['score'],
                 ['blue' if source == 'generated' else 'red' for source in master_updated_df['source']],
                 'weight',
                 'score')

    plot_lr_line(master_updated_df['smile'].apply(lambda x: len(x)),
                 master_updated_df['weight'],
                 ['blue' if source == 'generated' else 'red' for source in master_updated_df['source']],
                 'smiles_length',
                 'weight')

    """
    quantitative structure-activity relationship (QSAR) models - collectively referred 
    to as (Q)SARs - are mathematical models that can be used to predict the physicochemical, 
    biological and environmental fate properties of compounds from the knowledge of their chemical structure
    """
    _, smi2latent_model, lat2states_model, sample_model = load_autoencoder()

    st = SmilesTokenizer()
    vectorized = st.smiles2onehot(master_updated_df['smile'], MAX_LEN)
    latent = smi2latent_model.predict(vectorized[:, :-1, :])
    bindscore = np.array(master_updated_df['score'])

    # Train binding score model
    if not os.path.exists(BINDING_SCORE_MODEL_PATH):
        os.mkdir(BINDING_SCORE_MODEL_PATH)
        train_bindscore_model(latent, bindscore)

    # Load model and make predictions
    print("\n# Predict molecules binding score ...")
    if os.path.exists(BINDING_SCORE_MODEL_PATH):
        print("  Loading binding score model ...")
        bindscore_model = load_model_json(os.path.join(BINDING_SCORE_MODEL_PATH, 'binding_score_model.json'))
        bindscore_model.load_weights(os.path.join(BINDING_SCORE_MODEL_PATH, 'binding_score_model.h5'))

        print("  Predict binding scores ...")
        y_pred = bindscore_model.predict(latent)
        y_pred = y_pred.squeeze(axis=1)
        preds_df = pd.DataFrame({'smile': master_updated_df['smile'], 'score_pred': pd.Series(y_pred)})
        master_updated_df = pd.merge(master_updated_df, preds_df, on=['smile'], how='inner')
        master_updated_df.drop_duplicates(keep='first', inplace=True)
        print(master_updated_df.head(5))

        r = np.corrcoef(master_updated_df['score'], master_updated_df['score_pred'])[0, 1]
        print("  Correlation: %.3f" % r)

        # Generate low score molecules
        gen_df = gen_low_score_mols(latent,
                                    lat2states_model,
                                    sample_model,
                                    bindscore_model,
                                    master_updated_df,
                                    pc0_bounds=PC0_BOUNDS,
                                    pc1_bounds=PC1_BOUNDS)

        """
        Create fine tuning dataset. For the molecules from last iteration:
        1. Pick the best molecules by binding score from master table
        2. Pick best generated molecules by predicted binding score
        3. Generate similar molecules foreach molecule
        4. Remove duplicates
        """

        print("\n#  Create fine tuning dataset ...")

        # 1. Best real molecules
        print("  1. Pick best real molecules ...")
        top_master_df = master_updated_df[master_updated_df['score'] < MASTER_THRESH]
        top_master_df = top_master_df.sort_values('score_pred', ascending=True)
        top_master_smi = list(top_master_df['smile'])

        # 2. Best generated molecules
        print("  2. Pick best generated molecules")
        top_gen_df = gen_df[gen_df['score_pred'] < GEN_THRESH]
        top_gen_df = top_gen_df.sort_values('score_pred', ascending=True)

        check_quality_preds(list(top_gen_df['smile']), vectorized[:, :-1, :])
        top_gen_smi = list(set(top_gen_df['smile']))
        print("%d molecules without dups ..." % len(top_gen_smi))
        top_gen_smi_valid = filter_valid_mols(top_gen_smi, max_carbons=6)
        print("%d valid molecules remaining ..." % len(top_gen_smi_valid))

        # 3. Most similar molecules
        print("  3. Generate similar molecules ...")
        smi_list_nodups = list(set(top_master_smi + top_gen_smi_valid))

        # convert to one hot encoding and get latent representation
        similarity_factor = 0.95
        sim_smi_list = []
        for i, smi in tqdm(enumerate(smi_list_nodups)):
            vec = st.smiles2onehot([smi], MAX_LEN)
            lat = smi2latent_model.predict(vec[:, :-1, :])
            sim_lat = similarity_factor * lat
            sim_smi = latent2smiles(lat2states_model, sample_model, sim_lat, MAX_LEN)
            sim_smi_list.append(sim_smi)

        sim_smi_list = filter_valid_mols(sim_smi_list, max_carbons=6)

        ft_smi_list = list(set(top_master_smi + top_gen_smi_valid + sim_smi_list))
        print("Saving %d molecules for fine tuning ..." % (len(ft_smi_list)))
        f = open(os.path.join(BINDING_RESULTS_PATH, 'finetune_%s.smi' % GENERATION_ID), 'w')
        for smi in ft_smi_list:
            f.write(smi + '\n')
