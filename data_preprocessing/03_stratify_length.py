import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parse_config import ParseConfig

config = ParseConfig('../config.ini')

INPUT_SMI_FILE = "../data/dataset_cleansed.smi"
OUTPUT_SMI_FILE = "../data/dataset_cleansed_strat.smi"


if __name__ == '__main__':
    with open(INPUT_SMI_FILE, 'r') as f:
        smiles = [l.rstrip() for l in f]

    print(f'Input SMILES num: {len(smiles)}')

    len_list = [len(smi) for smi in smiles]

    # Get distribution by SMILE length
    fig, ax = plt.subplots()
    sns.distplot(len_list, kde=False, hist=True, norm_hist=False)
    plt.savefig('../data/length_dist_in.png', bbox_inches='tight')

    # We are going to train with a flatten length distribution so
    # that the probabilities of generating any random length are equal.
    # We are going to crop every length.
    len_dic_in = {}
    smiles_out = []

    for i in range(len(smiles)):
        if len_list[i] not in len_dic_in:
            len_dic_in[len_list[i]] = [smiles[i]]
        else:
            len_dic_in[len_list[i]].append(smiles[i])

    for length, smi_list_in in len_dic_in.items():
        if length < 60:
            num_samples = 0
            replace = False
        elif 60 <= length < 90:
            num_samples = 0
            replace = True
        elif 90 <= length < 100:
            num_samples = 4000
            replace = True
        elif 100 <= length < 110:
            num_samples = 3000
            replace = True
        else:
            num_samples = 2000
            replace = True

        smi_list_out = list(np.random.choice(smi_list_in, num_samples, replace=replace))
        smiles_out = smiles_out + smi_list_out

    print(f'Output SMILES num: {len(smiles_out)}')

    len_list = [len(smi) for smi in smiles_out]

    fig, ax = plt.subplots()
    sns.distplot(len_list, kde=False, hist=True, norm_hist=False)
    plt.savefig('../data/length_dist_out.png', bbox_inches='tight')

    with open(OUTPUT_SMI_FILE, 'w') as f:
        for smi in smiles_out:
            f.write(smi + '\n')
