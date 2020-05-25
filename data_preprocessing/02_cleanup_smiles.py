"""
Remove duplicates, salts and stereochemical information. In addition, pre-processing
filtered out nucleic acids and long peptides which lay outside of the chemical space from which we sought to
sample. The NN was ultimately trained on SMILES strings with lengths from 34 to 128 SMILES characters (tokens)
"""

from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from model.utils.smiles_tokenizer import SmilesTokenizer
from parse_config import ParseConfig
from model.utils.utils import filter_valid_mols

RDLogger.DisableLog('rdApp.*')

config = ParseConfig('../config.ini')

INPUT_CHEMBL_FILE = "../data/chembl_25.smi"
INPUT_MOSES_FILE = "../data/moses.smi"
OUTPUT_SMI_FILE = "../data/dataset_cleansed.smi"


class Preprocessor(object):
    def __init__(self):
        self.normalizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def process(self, smi):
        # Remove Si as PyRx does not support it
        if 'Si' in smi:
            return None

        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normalizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None


if __name__ == '__main__':
    pp = Preprocessor()

    with open(INPUT_CHEMBL_FILE, 'r') as f:
        chembl_smiles = [l.rstrip() for l in f]

    with open(INPUT_MOSES_FILE, 'r') as f:
        moses_smiles = [l.rstrip() for l in f]

    smiles = chembl_smiles + moses_smiles
    print(f'Input SMILES num: {len(smiles)}')

    print('\nStart to clean up ...')
    pp_smiles = [pp.process(smi) for smi in tqdm(smiles)]
    cl_smiles = list(set([s for s in tqdm(pp_smiles) if s]))
    print('Step 1 / 3 completed')

    print("\nFilter valid molecules ...")
    cl_smiles = filter_valid_mols(cl_smiles)
    print('Step 2 / 3 completed')

    # token limits
    print("\nKeeping molecules with lengths between %d and %d ..." % (config.min_len, config.max_len))
    out_smiles = []
    st = SmilesTokenizer()
    total = len(cl_smiles)
    count = 0
    skip_count = 0
    timeout_count = 0
    for cl_smi in tqdm(cl_smiles):
        try:
            tokenized_smi = st.tokenize(cl_smi)
            if not tokenized_smi:
                timeout_count += 1
            elif config.min_len <= len(tokenized_smi) <= config.max_len:
                out_smiles.append(cl_smi)
        except:
            skip_count += 1
        count += 1

    print('Step 3 / 3 completed')

    print(f'\nOutput SMILES num: {len(out_smiles)}')

    with open(OUTPUT_SMI_FILE, 'w') as f:
        for smi in out_smiles:
            f.write(smi + '\n')
