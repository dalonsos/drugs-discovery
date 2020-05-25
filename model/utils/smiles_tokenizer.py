import numpy as np
import time
from tqdm import tqdm


class SmilesTokenizer(object):
    def __init__(self):
        atoms = [
            'Li',
            'Na',
            'Al',
            'Si',
            'Cl',
            'Sc',
            'Zn',
            'As',
            'Se',
            'Br',
            'Sn',
            'Te',
            'Cn',
            'H',
            'B',
            'C',
            'N',
            'O',
            'F',
            'P',
            'S',
            'K',
            'V',
            'I',
        ]
        special = [
            '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's'
        ]
        padding = ['G', 'A', 'E']

        # List with all possible SMILES components: atoms, special characters and padding characters.
        # Padding characters are created by us: G (start of sequence), 'A' (padding to smiles_max_length) and
        # E (end of sequence)
        self.table = sorted(atoms, key=len, reverse=True) + special + padding
        self.table_len = len(self.table)  # 52 tokens

        self.one_hot_dict = {}
        for i, symbol in enumerate(self.table):
            vec = np.zeros(self.table_len, dtype=np.float32)
            vec[i] = 1
            self.one_hot_dict[symbol] = vec

        self.char_to_int = {k: list(v).index(1.0) for k, v in self.one_hot_dict.items()}
        self.int_to_char = {v: k for k, v in self.char_to_int.items()}

    def tokenize(self, smiles):
        """
        Converts SMILE string to tokens list making use of tokens list.
        I.e: 'LiNA(C)' is converted to ['Li', 'Na', '(', 'C', ')']
        """
        N = len(smiles)
        i = 0
        token = []

        timeout = time.time() + 5   # 5 seconds from now
        while i < N:
            for j in range(self.table_len):
                symbol = self.table[j]
                if symbol == smiles[i:i + len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
            if time.time() > timeout:
                break
        return token

    @staticmethod
    def pad(tokenized_smiles, max_len):
        """
        Pads a tokenized SMILE with go (G), end (E) and empty (A) characters
        until max_len. I.e. If tokenized SMILE is ['Li', 'Na', '(', 'C', ')']
        and max_len = 10 then the result is
        ['G', 'Li', 'Na', '(', 'C', ')', 'E', 'A', 'A', 'A']
        """
        return ['G'] + tokenized_smiles + ['E'] + ['A' for _ in range(max_len - len(tokenized_smiles))]

    def one_hot_encode(self, tokenized_smiles):
        """
        Converts a tokenized SMILE to a one hot vector.
        Vocabulary size is 52 so it the tokenized SMILE has length N then
        result shape is (1, N, 52)
        """
        result = np.array(
            [self.one_hot_dict[symbol] for symbol in tokenized_smiles],
            dtype=np.uint8)
        result = result.reshape(1, result.shape[0], result.shape[1])
        return result

    def smiles2onehot(self, smiles, max_len):
        print('\n# Tokenizing SMILES ...')
        tokenized_smiles = [self.tokenize(smi) for smi in tqdm(smiles)]
        tokenized_smiles_len = [len(tokenized_smi) for tokenized_smi in tokenized_smiles]
        print('Min/Max length: %d, %d' % (min(tokenized_smiles_len), max(tokenized_smiles_len)))

        print('\n# Padding tokenized SMILES ...')
        tokenized_pad_smiles = [self.pad(tokenized_smi, max_len) for tokenized_smi in tqdm(tokenized_smiles)]
        tokenized_pad_smiles_len = [len(tokenized_smi) for tokenized_smi in tqdm(tokenized_pad_smiles)]
        print('Min/Max length: %d, %d' % (min(tokenized_pad_smiles_len), max(tokenized_pad_smiles_len)))

        print("\n# One hot encoding ...")
        vectorized = np.zeros(shape=(len(smiles), max_len + 2, self.table_len), dtype=np.uint8)
        for i, tok_smi in tqdm(enumerate(tokenized_pad_smiles)):
            vectorized[i] = self.one_hot_encode(tok_smi)

        return vectorized

    def onehot2smiles(self, vectorized):
        """
        Converts a one hot tensor back to a smiles list
        without the padding characters
        """
        smiles_list = []
        for i in range(vectorized.shape[0]):
            smiles = "".join([self.int_to_char[idx] for idx in np.argmax(vectorized[i], axis=1)])
            smiles = "".join([c for c in smiles if c not in ['G', 'E', 'A']])
            smiles_list.append(smiles)

        return smiles_list
