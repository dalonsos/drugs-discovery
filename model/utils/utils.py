import os
import json
from rdkit import RDLogger, Chem
from difflib import ndiff
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
from keras.models import model_from_json
import matplotlib.pyplot as plt
from model.utils.smiles_tokenizer import SmilesTokenizer

RDLogger.DisableLog('rdApp.*')


def load_model_json(json_path):
    """
    :param json_path: json file that contains the compiled model with its architecture
    """
    with open(json_path, 'r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)

    return model


def get_best_model_name(model_name, loss_csv):
    """
    Returns the name of the weights file .h5 with the least
    validation loss after training
    :param loss_csv: csv_logger callback result
    :param model_name: model name in config file
    """
    log_df = pd.read_csv(loss_csv, sep=';')
    min_log_df = log_df[log_df['val_loss'] == min(log_df['val_loss'])]
    min_log_df = min_log_df[min_log_df['loss'] == min(min_log_df['loss'])]
    epoch = str(min_log_df['epoch'].values[0])
    if len(epoch) == 1:
        epoch = '0' + epoch

    checkpoint_name = "%s-%s-%.3f.h5" % (model_name,
                                         epoch,
                                         min_log_df['val_loss'].values[0])

    return checkpoint_name


def filter_valid_mols(smiles_list, max_carbons=None):
    """
    Filters a list of smiles an returns a list
    with the valid ones
    :param smiles_list: smiles list
    :param max_carbons: max number of consecutive C allowed
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    valid_idxs = [i for i, mol in enumerate(mols) if mol is not None]
    smiles_valid = [smiles_list[i] for i in valid_idxs]
    print("%d invalid molecules removed" % (len(smiles_list) - len(smiles_valid)))

    if max_carbons is not None:
        smiles_valid = [smi for smi in smiles_valid if 'C' * (max_carbons + 1) not in smi]

    return smiles_valid


def check_quality_preds(smiles_pred, x_train):
    """
    Given a list with smiles predictions (no padding) and
    the training tensor it checks:
        - Validity: ratio of valid smiles
        - Uniqueness: ratio of not repeated valid smiles
        - Originality: new valid generated smiles to training smiles ratio
    """
    print("\nChecking validity, uniqueness and originality of predictions ...")
    smiles_pred_valid = filter_valid_mols(smiles_pred)
    validity = len(smiles_pred_valid) / len(smiles_pred)
    print('Validity:', f'{validity:.2%}')

    st = SmilesTokenizer()
    smiles_train = st.onehot2smiles(x_train)
    smiles_train_set = set(smiles_train)
    original = []
    for smile in smiles_pred_valid:
        if smile not in smiles_train_set:
            original.append(smile)

    if len(smiles_pred_valid) != 0:
        uniqueness = len(set(smiles_pred_valid)) / len(smiles_pred_valid)
        originality = len(set(original)) / len(set(smiles_pred_valid))
        print('Uniqueness:', f'{uniqueness:.2%}')
        print('Originality: ', f'{originality:.2%}')
    else:
        uniqueness = 0
        originality = 0
        print('Uniqueness: 0.00%')
        print('Originality: 0.00%')

    return validity, uniqueness, originality


def levenshtein_distance(str1, str2):
    """
    Levenshtein distance between 2 strings
    """
    counter = {"+": 0, "-": 0}
    distance = 0
    for edit_code, *_ in ndiff(str1, str2):
        if edit_code == " ":
            distance += max(counter.values())
            counter = {"+": 0, "-": 0}
        else:
            counter[edit_code] += 1
    distance += max(counter.values())

    return distance


def plot_2d_distribution(sample, save_path, points_dic={}):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    x = sample[:, 0]
    y = sample[:, 1]

    x_lims = (np.percentile(x, 1, axis=0),
              np.percentile(x, 99, axis=0))

    y_lims = (np.percentile(y, 1, axis=0),
              np.percentile(y, 99, axis=0))

    # Hist
    ax1.hist2d(x, y, bins=50, cmap='Blues')
    if points_dic:
        for r, point_list in points_dic.items():
            c = np.random.rand(3)
            for point in point_list:
                ax1.scatter(point[0], point[1], s=17, c=c)
    ax1.set_xlim([x_lims[0], x_lims[1]])
    ax1.set_ylim([y_lims[0], y_lims[1]])
    ax1.set_xlabel('PC0')
    ax1.set_ylabel('PC1')

    # kde
    xx, yy = np.mgrid[x_lims[0]:x_lims[1]:100j, y_lims[0]:y_lims[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    ax2.set_xlim(x_lims[0], x_lims[1])
    ax2.set_ylim(y_lims[0], y_lims[1])
    # Contourf plot
    cfset = ax2.contourf(xx, yy, f, cmap='Blues')
    cset = ax2.contour(xx, yy, f, colors='k')
    ax2.clabel(cset, inline=1, fontsize=10)
    ax2.set_xlabel('PC0')
    ax2.set_ylabel('PC1')

    plt.savefig(save_path, bbox_inches='tight')


def latent2smiles(lat2states_model,
                  sample_model,
                  latent_tensor,
                  max_len,
                  temperature=1.0):
    """
    Generates a smiles sample from latent space
    :param lat2states_model: latent to states model
    :param sample_model: sample model
    :param latent_tensor: a 2D tensor representing the latent space
    :param max_len: max length of the generated sequence
    :param temperature: higher temperatures make your output more random
    """
    # https://stackoverflow.com/questions/54030842/character-lstm-keeps-generating-same-character-sequence

    # Decode states and set reset the LSTM cells with them
    states = lat2states_model.predict(latent_tensor)
    sample_model.layers[1].reset_states(states=[states[0], states[1]])
    # Prepare the input char
    st = SmilesTokenizer()
    start_idx = st.char_to_int["G"]
    sample_vec = np.zeros((1, 1, st.table_len))
    sample_vec[0, 0, start_idx] = 1
    smiles_str = ""
    # Loop and predict next char
    for i in range(max_len):
        # code to sample an index from a probability array
        preds = sample_model.predict(sample_vec)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        sample_idx = np.argmax(preds)

        # get next predicted character from index
        sample_char = st.int_to_char[sample_idx]
        if sample_char != "E":
            smiles_str = smiles_str + st.int_to_char[sample_idx]
            sample_vec = np.zeros((1, 1, st.table_len))
            sample_vec[0, 0, sample_idx] = 1
        else:
            break

    return smiles_str


