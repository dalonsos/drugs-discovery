""""
Evaluating each molecule's binding affinity with the coronavirus protease via PyRx
is a lengthy process with ~2 molecules per minute. Running an analysis of 1.5k molecules
was therefore not possible as that would take over 12.5 hours.

In order to minimize time the function initialize_generation_from_mols() randomly picks 30 molecules,
then iterates through the rest of the list calculating that molecules Tanimoto similarity scores to the
molecules so far added to the list, and only 'accepts' the molecule if the maximum similarity score
is less than a certain threshold. This ensures that even a smaller sample will feature a diverse set of molecules
"""

import os
from tqdm import tqdm
import random
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import PropertyMol, Descriptors
from rdkit.Chem.QED import qed
from model.utils.utils import *
from parse_config import ParseConfig


RDLogger.DisableLog('rdApp.*')


def initialize_generation_from_mols(generated_smiles_path,
                                    desired_length,
                                    initial_length=30):
    """
    It is assumed that similar molecules will have similar docking scores,
    so we want a diverse sample of molecules

    Given a list of smiles as input, randomly shuffles them, adds the first n molecules and
    then sets a max-similarity threshold between any new molecule and the existing list.
    Iteratively increases the threshold until N molecules are picked. This method ensures
    diveristy in the sample
    """

    print("\n# Initialize generation from %d molecules up to %d ..." % (initial_length, desired_length))

    # desired_length must be greater than initial length
    assert desired_length > initial_length

    with open(generated_smiles_path) as file:
        smiles_list = [line.strip() for line in file]

    mols_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    random.shuffle(mols_list)

    # prepare fingerprints for similarity calculations
    mol_fingerprints = []
    for mol in mols_list:
        mol_fingerprints.append(Chem.RDKFingerprint(mol))

    selected_mols = mols_list[0:initial_length]
    selected_fingerprints = mol_fingerprints[0:initial_length]
    remaining_mols = mols_list[initial_length:]
    remaining_fingerprints = mol_fingerprints[initial_length:]

    # start similarity threshold on 0.05.
    # if similarity is below 0.05 the molecules are considered different
    thresh_list = []
    length_list = []
    similarity_threshold = 0.05
    # keep iterating until selected_mols list has desired_length
    while (len(selected_mols) < desired_length) and similarity_threshold <= 1.0:
        print("  Current threshold: %.2f, Currenth length: %d" % (round(similarity_threshold, 2), len(selected_mols)))
        for fingerprint, mol in zip(remaining_fingerprints, remaining_mols):
            max_similarity = np.max(DataStructs.BulkTanimotoSimilarity(fingerprint, selected_fingerprints))
            if (max_similarity <= similarity_threshold) and (max_similarity < 1):
                selected_fingerprints.append(fingerprint)
                selected_mols.append(mol)

        thresh_list.append(similarity_threshold)
        length_list.append(len(selected_mols))
        # lossen up similarity threshold
        similarity_threshold += 0.05

    return selected_mols


def iterate_alpha(alpha_code):
    """
    In the PyRx GUI molecule names would sort oddly
    in any numeric order, so it is needed to order
    molecules by a four letter code. This function
    iterates the four letter code

    Generates the next sequence in a four characters code.
    For example: iterate_alpha('AAAA') returns 'AAAB'
    """
    numbers = []
    for letter in alpha_code:
        number = ord(letter)
        numbers.append(number)

    if numbers[3] + 1 > 90:
        if numbers[2] + 1 > 90:
            if numbers[1] + 1 > 90:
                if numbers[0] + 1 > 90:
                    raise ValueError('Too long for alpha code')
                else:
                    numbers[3] = 65
                    numbers[2] = 65
                    numbers[1] = 65
                    numbers[0] = numbers[0] + 1
            else:
                numbers[3] = 65
                numbers[2] = 65
                numbers[1] = numbers[1] + 1
        else:
            numbers[3] = 65
            numbers[2] = numbers[2] + 1
    else:
        numbers[3] = numbers[3] + 1

    new_code = ""
    for number in numbers:
        new_code += chr(number)

    return new_code


def append_to_tracking_table(mols_to_append,
                             source,
                             generation,
                             master_table_path,
                             train_data_path):
    """
    This function adds new rows to a pandas dataframe with the following columns:
        - id_code: string with an id with 4 characters. If the table
                   already has entries, then the next codes are computed. I.e:
                   if the last entry code is AAAB then the new entry code will
                   be AAAC
        - generation: number associated to the genetic algorithm generation
        - smile: SMILE representation
        - source: descriptive string indicating the source of the molecule
        - weight: molecular weight
        - logp: logP value which provides indications on whether a substance will be absorbed by plants, animals,
                humans, or other living tissue; or be easily carried away and disseminated by water
        - qed: quantitative estimation of drug-likeness
        - score: binding score
    """

    print("\n# Append new molecules to tracking table ...")

    mols_to_export = []
    rows_list = []

    # Check if master table exists. Otherwise initialize it
    if os.path.exists(os.path.join(master_table_path, 'master_table.csv')):
        master_table = pd.read_csv(os.path.join(master_table_path, 'master_table.csv'),
                                   sep=',',
                                   dtype={'gen': 'str'})

    else:
        master_table = pd.DataFrame(columns=['id', 'gen', 'smile', 'source', 'weight', 'logp', 'qed', 'score'])

    # Filters the corresponding generation
    master_table_gen = master_table[master_table['gen'] == generation]
    if master_table_gen.shape[0] == 0:
        # If no entries corresponding to the generation
        id_code = 'AAAA'
    else:
        # Compute the next id_code
        master_table_gen_ids = master_table_gen.sort_values('id', ascending=True)
        master_table_gen_max_id = master_table_gen_ids.tail(1)
        key = master_table_gen_max_id['id'].keys()[0]
        id_code = iterate_alpha(str(master_table_gen_max_id['id'][key]))

    # Get all training smiles
    training_data = pd.read_csv(train_data_path, header=None)
    training_set = set(list(training_data[0]))

    for mol in mols_to_append:
        # Set molecule title property for tracking in PyRx
        # and append molecule to list
        pm = PropertyMol.PropertyMol(mol)
        title = 'id_' + str(id_code) + '_gen_' + generation
        pm.SetProp('Title', title)
        mols_to_export.append(pm)

        # Update pandas dataframe with new molecule
        try:
            mol_dict = {}
            mol_dict['id'] = id_code
            mol_dict['gen'] = generation
            smile = Chem.MolToSmiles(mol)
            mol_dict['smile'] = smile

            if (source != 'hiv' and source != 'manual' and source != 'baseline') and (smile in training_set):
                # if molecule is in training set
                mol_dict['source'] = 'training'
            else:
                mol_dict['source'] = source

            mol_dict['weight'] = Descriptors.MolWt(mol)
            mol_dict['logp'] = Descriptors.MolLogP(mol)
            mol_dict['qed'] = qed(mol)
            mol_dict['score'] = 99.9
        except:
            continue

        rows_list.append(mol_dict)
        id_code = iterate_alpha(id_code)

    mols_export = pd.DataFrame(rows_list)
    master_table = master_table.append(mols_export)

    print("  %d new molecules were added to master table" % len(mols_export))
    print("  %d rows in master table" % len(master_table))

    print("  Saving tracking table updated ...")
    master_table.to_csv(os.path.join(master_table_path, 'master_table.csv'), sep=',', index=False)

    return mols_to_export, master_table


def write_gen_to_sdf(mols_export,
                     generation,
                     batch_size,
                     sdf_output_path):
    """
    Generates a .sdf file with a set of molecules. If the set of molecules
    is greater than batch_size, several .sdf files will be created
    :param mols_export: list of molecules to export to .sdf
    :param generation: generation id
    :param batch_size: max number of molecules per file. several
                       files will be generated if the number of molecules
                       to export is greater than this parameter
    :param sdf_output_path: output path where .sdf file will be generated
    """

    print("\n# Generate sdf file with %d new molecules ..." % len(mols_export))
    if len(mols_export) > batch_size:
        batches = (len(mols_export) // batch_size) + 1
        for i in tqdm(range(batches)):
            batch_to_export = mols_export[i * batch_size:(i + 1) * batch_size]
            w = Chem.SDWriter(os.path.join(sdf_output_path, 'gen_' + str(generation) + '_batch_' + str(i + 1) + '.sdf'))
            for m in batch_to_export:
                w.write(m)
    else:
        w = Chem.SDWriter(os.path.join(sdf_output_path, 'gen_' + str(generation) + '.sdf'))
        for m in tqdm(mols_export):
            w.write(m)

    # Noticed an issue where the very last line item of an sdf write
    # is not written correctly until another arbitary write is made
    w = Chem.SDWriter(os.path.join(sdf_output_path, 'junk.sdf'))
    w.write(m)
    w.close()


if __name__ == '__main__':
    config = ParseConfig('../config.ini')

    # Parameters
    ADD_MANUAL_MOLS = False
    ADD_TRAIN_MOLS = False
    NUM_GEN_MOLS = 20
    NUM_TRAIN_MOLS = 400
    TRAIN_DATA_PATH = '../data/dataset_cleansed.smi'
    SDF_PATH = '../data/generations/sdf_files'
    MASTER_TABLE_PATH = '../data/generations'
    gen_files = os.listdir('gen_results/%s/data' % config.model_name)
    gen_files = [f for f in gen_files if f.startswith('generated_')]
    last_gen_file = sorted(gen_files)[-1]
    GENERATION_ID = last_gen_file.split('_')[-1].split('.')[0]
    GEN_SMILES_PATH = 'gen_results/%s/data/generated_%s.smi' % (config.model_name, GENERATION_ID)

    # Create folders
    if not os.path.exists(os.path.abspath('../data/generations')):
        os.mkdir(os.path.abspath('../data/generations'))

    for folder in ['sdf_files', 'binding_results']:
        if not os.path.exists(os.path.join('../data/generations', folder)):
            os.mkdir(os.path.join('../data/generations', folder))

    if not os.path.exists(os.path.join('../data/generations/binding_results', 'gen_%s' % GENERATION_ID)):
        os.mkdir(os.path.join('../data/generations/binding_results', 'gen_%s' % GENERATION_ID))

    # Initialize new molecules and update master table
    mols_export = []

    gen_mols = initialize_generation_from_mols(GEN_SMILES_PATH, NUM_GEN_MOLS, initial_length=10)
    new_mols_to_test, _ = append_to_tracking_table(gen_mols,
                                                   'generated',
                                                   GENERATION_ID,
                                                   MASTER_TABLE_PATH,
                                                   TRAIN_DATA_PATH)
    mols_export = mols_export + new_mols_to_test

    if ADD_TRAIN_MOLS:
        training_data = pd.read_csv(TRAIN_DATA_PATH, header=None)
        training_smiles = list(set(list(training_data[0])))
        idxs = np.random.choice([i for i in range(len(training_smiles))], NUM_TRAIN_MOLS, replace=False)
        training_mols_sample = []
        training_smis_sample = []
        for i in idxs:
            training_smis_sample.append(training_smiles[i])
            training_mols_sample.append(Chem.MolFromSmiles(training_smiles[i]))

        new_mols_to_test, _ = append_to_tracking_table(training_mols_sample,
                                                       'training',
                                                       GENERATION_ID,
                                                       MASTER_TABLE_PATH,
                                                       TRAIN_DATA_PATH)
        mols_export = mols_export + new_mols_to_test

    if ADD_MANUAL_MOLS:
        # Add manually HIV inhibitors
        with open('../data/hiv_inhibitors_cleansed.smi') as file:
            hiv_smiles = [line.strip() for line in file]
        hiv_smiles = filter_valid_mols(hiv_smiles)
        hiv_mols = [Chem.MolFromSmiles(smi) for smi in hiv_smiles]
        new_mols_to_test, _ = append_to_tracking_table(hiv_mols,
                                                       'hiv',
                                                       GENERATION_ID,
                                                       MASTER_TABLE_PATH,
                                                       TRAIN_DATA_PATH)
        mols_export = mols_export + new_mols_to_test

        # Add manually remdesivir
        with open('../data/remdesivir.smi') as file:
            remdesivir_smile = [line.strip() for line in file]
        remdesivir_smile = filter_valid_mols(remdesivir_smile)
        remdesivir_mol = [Chem.MolFromSmiles(smi) for smi in remdesivir_smile]
        new_mols_to_test, _ = append_to_tracking_table(remdesivir_mol,
                                                       'remdesivir',
                                                       GENERATION_ID,
                                                       MASTER_TABLE_PATH,
                                                       TRAIN_DATA_PATH)
        mols_export = mols_export + new_mols_to_test

        # Add manually candidates
        with open('../data/candidates_cleansed.smi') as file:
            candidates_smiles = [line.strip() for line in file]
        candidates_smiles = filter_valid_mols(candidates_smiles)
        candidates_mols = [Chem.MolFromSmiles(smi) for smi in candidates_smiles]
        new_mols_to_test, _ = append_to_tracking_table(candidates_mols,
                                                       'candidates',
                                                       GENERATION_ID,
                                                       MASTER_TABLE_PATH,
                                                       TRAIN_DATA_PATH)
        mols_export = mols_export + new_mols_to_test

        # Add manually high score molecules
        with open('../data/high_score_cleansed.smi') as file:
            high_score_smiles = [line.strip() for line in file]
        high_score_smiles = filter_valid_mols(high_score_smiles)
        high_score_mols = [Chem.MolFromSmiles(smi) for smi in high_score_smiles]
        new_mols_to_test, _ = append_to_tracking_table(high_score_mols,
                                                       'high_score',
                                                       GENERATION_ID,
                                                       MASTER_TABLE_PATH,
                                                       TRAIN_DATA_PATH)
        mols_export = mols_export + new_mols_to_test

    # Create SDF file
    write_gen_to_sdf(mols_export,
                     GENERATION_ID,
                     len(mols_export),
                     SDF_PATH)
