import os
import pickle
from tqdm import tqdm
import seaborn as sns
from rdkit.Chem.QED import qed
from rdkit.Chem import Draw
import silence_tensorflow.auto
import tensorflow as tf
from sklearn.decomposition import PCA
from model.utils.utils import *
from model.utils.smiles_tokenizer import SmilesTokenizer
from parse_config import ParseConfig

RDLogger.DisableLog('rdApp.*')


class Generator(object):
    def __init__(self):
        self.config = ParseConfig('../config.ini')
        self.st = SmilesTokenizer()

        print('TF version is %s ...\n' % tf.__version__)
        print('Checking if TF was built with CUDA: %s ...\n' % tf.test.is_built_with_cuda())

        if not self.config.tf_use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if tf.test.gpu_device_name():
            print('GPU found')
            print(tf.config.list_physical_devices('GPU'), '\n')
        else:
            print("No GPU found. Running on CPU")

        # Get folder
        if not os.path.exists('train_results/%s/finetune_results' % self.config.model_name):
            self.ft_index = '00'
            self.folder = os.path.abspath('train_results/%s/' % self.config.model_name)
        else:
            folders = os.listdir('train_results/%s/finetune_results' % self.config.model_name)
            ft_folders = [f for f in folders if f.startswith('ft_')]
            if len(ft_folders) == 0:
                self.ft_index = '00'
                self.folder = os.path.abspath('train_results/%s/' % self.config.model_name)
            else:
                last_ft_folder = sorted(ft_folders)[-1]
                self.ft_index = last_ft_folder.split('_')[-1]
                self.folder = 'train_results/%s/finetune_results/%s' % (self.config.model_name, last_ft_folder)

        # Load models
        self.checkpoint_name = get_best_model_name(self.config.model_name,
                                                   os.path.join(self.folder, 'loss.csv'))

        print("\n# Loading models from checkpoint '%s' ..." % self.checkpoint_name)
        self.lstm_autoencoder_model = load_model_json(os.path.join(self.folder, 'models/lstm_autoencoder_model.json'))
        self.lstm_autoencoder_model.load_weights(os.path.join(self.folder, 'checkpoints/' + self.checkpoint_name))

        self.smi2latent_model = load_model_json(os.path.join(self.folder, 'models/smi2latent_model.json'))
        # transfer weights
        for i in range(1, 4):
            self.smi2latent_model.layers[i].set_weights(self.lstm_autoencoder_model.layers[i].get_weights())

        self.lat2states_model = load_model_json(os.path.join(self.folder, 'models/lat2states_model.json'))
        # transfer weights
        for i in range(1, 3):
            self.lat2states_model.layers[i].set_weights(self.lstm_autoencoder_model.layers[i + 4].get_weights())

        self.sample_model = load_model_json(os.path.join(self.folder, 'models/sample_model.json'))
        # transfer weights
        for i in range(1, 3):
            self.sample_model.layers[i].set_weights(self.lstm_autoencoder_model.layers[i + 6].get_weights())

        # Create generator results folder
        print("\n# Creating generator folder ...")
        if not os.path.exists(os.path.abspath('gen_results')):
            os.mkdir(os.path.abspath('gen_results'))

        self.gen_folder = 'gen_results/%s/' % self.config.model_name
        if not os.path.exists(os.path.abspath(self.gen_folder)):
            os.mkdir(os.path.abspath(self.gen_folder))

        for folder in ['data', 'plots', 'images']:
            if not os.path.exists(os.path.join(self.gen_folder, folder)):
                os.mkdir(os.path.abspath(os.path.join(self.gen_folder, folder)))

        print("\n# Getting latent representation ...")
        with open(os.path.join(self.folder, 'data/x_sample_big.pickle'), 'rb') as f:
            self.x_sample = pickle.load(f)

        self.latent = self.smi2latent_model.predict(self.x_sample)
        self.smiles_list = self.st.onehot2smiles(self.x_sample)

    def gen_new_mols(self, num_mols=1000, qed_thresh=0.5, max_carbons=6):
        print("\n# Generating new molecules ...")

        np.random.seed(self.config.seed)

        # Get 2D latent representation
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent)
        print(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_))

        # Sample latent 2D representation
        num_mols = min(num_mols, latent_2d.shape[0])
        idxs = np.random.choice([i for i in range(latent_2d.shape[0])], num_mols, replace=False)
        latent_2d_sampled = latent_2d[idxs, :]
        plot_2d_distribution(latent_2d_sampled, os.path.join(self.gen_folder, 'plots/latent_2d_dist.png'))

        # Approximate latent 2D representation by a gaussian
        mean = np.mean(latent_2d_sampled, axis=0)
        print(mean)

        covariance = np.cov(latent_2d_sampled.T)
        print(covariance)

        latent_2d_gen = np.random.multivariate_normal(mean, covariance, num_mols)
        plot_2d_distribution(latent_2d_gen, os.path.join(self.gen_folder, 'plots/latent_2d_gen_dist.png'))
        for i in range(10, 100, 10):
            per = np.percentile(latent_2d_gen, i, axis=0)
            print("percentile %d ... PC0: %.4f, PC1: %.4f" % (i, per[0], per[1]))

        # Convert back generated 2D latent data to its original dimension
        latent_gen = pca.inverse_transform(latent_2d_gen)

        # Generate valid smiles from latent
        smiles_gen = [latent2smiles(self.lat2states_model, self.sample_model, latent_gen[i:i + 1], self.config.max_len)
                      for i in tqdm(range(latent_gen.shape[0]))]
        check_quality_preds(smiles_gen, self.x_sample)
        smiles_gen = list(set(smiles_gen))
        print("%d molecules without dups ..." % len(smiles_gen))
        smiles_gen_valid = filter_valid_mols(smiles_gen, max_carbons=max_carbons)
        print("%d valid molecules remaining ..." % len(smiles_gen_valid))

        # Filter molecules with QED higher than a threshold
        fig, ax = plt.subplots()
        qed_list = []
        smiles_gen_valid_filt = []
        for smi in smiles_gen_valid:
            try:
                qed_val = qed(Chem.MolFromSmiles(smi))
                qed_list.append(qed_val)
                if qed_thresh is not None and (0 < qed_thresh < 1) and (qed_val > qed_thresh):
                    smiles_gen_valid_filt.append(smi)
            except:
                continue
        sns.kdeplot(qed_list, ax=ax)
        plt.savefig(os.path.join(self.gen_folder, 'plots/qed_distribution.png'), bbox_inches='tight')
        if qed_thresh is not None and (0 < qed_thresh < 1):
            smiles_gen_valid = smiles_gen_valid_filt.copy()
            print("%d valid molecules with QED > %.2f ..." % (len(smiles_gen_valid), qed_thresh))

        # Save smiles
        f = open(os.path.join(self.gen_folder, 'data/generated_%s.smi' % self.ft_index), 'w')
        for smi in smiles_gen_valid:
            f.write(smi + '\n')

        f.close()

        # Plot some images
        idxs = np.random.choice([i for i in range(len(smiles_gen_valid))], 9, replace=False)

        mols_sample = []
        smis_sample = []
        for i in idxs:
            smis_sample.append(smiles_gen_valid[i])
            mols_sample.append(Chem.MolFromSmiles(smiles_gen_valid[i]))

        img = Draw.MolsToGridImage(mols_sample, molsPerRow=3, subImgSize=(400, 200))
        img.save(os.path.join(self.gen_folder, 'images/generated_mols.png'))

    def similarity_test(self):
        """
        Similar molecules produces similar fingerprints.
        To see if similar molecules produce similar vectors in the latent space, a simple
        search for similar molecules can be performed. Here the absolute
        difference between the latent vectors is used as a metric of similarity
        """

        print("\n# Similarity test ...")

        if not os.path.exists(os.path.join(self.gen_folder, 'similarity_test')):
            os.mkdir(os.path.join(self.gen_folder, 'similarity_test'))

        # Select random smile vector, get its fingerprint and calculate
        # for every smiles vector the absolute difference with it in the latent space.
        # Finally order the differences vector
        idx = np.random.choice([i for i in range(self.x_sample.shape[0])], 1, replace=False)[0]
        latent_mol = self.smi2latent_model.predict(self.x_sample[idx:idx + 1])
        sorti = np.argsort(np.sum(np.abs(self.latent - latent_mol), axis=1))

        with open(os.path.join(self.gen_folder, 'similarity_test/results.txt'), 'w') as f:
            f.write("Base molecule:\n")
            f.write("  %s\n\n" % self.smiles_list[idx])

            f.write("Most similar:\n")
            for smi in list(np.array(self.smiles_list)[sorti[1:10]]):
                f.write("  %s\n" % smi)
            f.write("\n")

            f.write("Most different:\n")
            for smi in list(np.array(self.smiles_list)[sorti[-9:]]):
                f.write("  %s\n" % smi)

        # Base molecule
        img = Draw.MolsToGridImage([Chem.MolFromSmiles(self.smiles_list[idx])], subImgSize=(400, 200))
        img.save(os.path.join(self.gen_folder, 'similarity_test/base_mol.png'))

        # Most similar
        img = Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in list(np.array(self.smiles_list)[sorti[1:10]])],
                                   molsPerRow=3,
                                   subImgSize=(400, 200))
        img.save(os.path.join(self.gen_folder, 'similarity_test/most_similar.png'))

        # Most different
        img = Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in list(np.array(self.smiles_list)[sorti[-9:]])],
                                   molsPerRow=3,
                                   subImgSize=(400, 200))
        img.save(os.path.join(self.gen_folder, 'similarity_test/most_different.png'))

    def interpolation_test(self):
        """
        In order to check if the latent space has a good structure, interpolation it's a good technique.
        Given two SMILES, a linear combination of both will be done in the
        continuous latent space so that the generated molecule is in between them
        """
        print("\n# Interpolation test ...")

        if not os.path.exists(os.path.join(self.gen_folder, 'interpolation_test')):
            os.mkdir(os.path.join(self.gen_folder, 'interpolation_test'))

        # Select 2 molecules
        idxs = np.random.choice([i for i in range(self.x_sample.shape[0])], 2, replace=False)

        smi0 = "".join(
            [self.st.int_to_char[idx] for
             idx in np.argmax(np.squeeze(self.x_sample[idxs[0]:idxs[0] + 1], axis=0), axis=1)])
        smi0 = "".join([c for c in smi0 if c not in ['G', 'E', 'A']])
        mol0 = Chem.MolFromSmiles(smi0)

        smi1 = "".join(
            [self.st.int_to_char[idx] for
             idx in np.argmax(np.squeeze(self.x_sample[idxs[1]:idxs[1] + 1], axis=0), axis=1)])
        smi1 = "".join([c for c in smi1 if c not in ['G', 'E', 'A']])
        mol1 = Chem.MolFromSmiles(smi1)

        latent0 = self.smi2latent_model.predict(self.x_sample[idxs[0]:idxs[0] + 1])
        latent1 = self.smi2latent_model.predict(self.x_sample[idxs[1]:idxs[1] + 1])
        mols = []
        smis = []
        rs = []
        rs_val = []
        ds = []
        valid = []
        ratios = np.linspace(0, 1, 11)
        for r in ratios:
            rlatent = (1.0 - r) * latent0 + r * latent1
            smi = latent2smiles(self.lat2states_model, self.sample_model, rlatent, self.config.max_len)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # if not None, mol is valid
                mols.append(mol)
                valid.append('valid: OK')
                rs_val.append(r)
            else:
                valid.append('valid: KO')

            smis.append(smi)
            rs.append(r)
            ds.append((levenshtein_distance(smi0, smi),
                       levenshtein_distance(smi1, smi)))

        with open(os.path.join(self.gen_folder, 'interpolation_test/results.txt'), 'w') as f:
            f.write("Base molecules\n")
            f.write("  [0] %s\n" % smi0)
            f.write("  [1] %s\n" % smi1)
            f.write("\n")

            f.write("Interpolations\n")
            for i in range(len(smis)):
                f.write("  [r: %.2f] [%s] [d: (%d,%d)] %s\n" % (rs[i], valid[i], ds[i][0], ds[i][1], smis[i]))

        img = Draw.MolsToGridImage([mol0, mol1], subImgSize=(400, 200), legends=['0', '1'])
        img.save(os.path.join(self.gen_folder, 'interpolation_test/base_mols.png'))

        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(400, 200), legends=[str(round(r, 2)) for r in rs_val])
            img.save(os.path.join(self.gen_folder, 'interpolation_test/interpolations.png'))
        else:
            os.remove(os.path.join(self.gen_folder, 'interpolation_test/interpolations.png'))

    def noise_test(self):
        """
        Check if novel molecules more or less similar to a lead molecule can be by adding
        a bit of randomness to an existing latent vector
        """
        print("\n# Noise test ...")

        if not os.path.exists(os.path.join(self.gen_folder, 'noise_test')):
            os.mkdir(os.path.join(self.gen_folder, 'noise_test'))

        idx = np.random.choice([i for i in range(self.x_sample.shape[0])], 1, replace=False)[0]

        smi_base = "".join(
            [self.st.int_to_char[idx] for
             idx in np.argmax(np.squeeze(self.x_sample[idx:idx + 1], axis=0), axis=1)])
        smi_base = "".join([c for c in smi_base if c not in ['G', 'E', 'A']])
        mol_base = Chem.MolFromSmiles(smi_base)

        latent_base = self.smi2latent_model.predict(self.x_sample[idx:idx + 1])

        scale = 0.40
        mols = []
        smis = []
        for i in range(20):
            latent_r = latent_base + scale * (np.random.randn(latent_base.shape[1]))
            smi = latent2smiles(self.lat2states_model, self.sample_model, latent_r, self.config.max_len)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
                smis.append(smi)

        with open(os.path.join(self.gen_folder, 'noise_test/results.txt'), 'w') as f:
            f.write("Base molecule\n")
            f.write("  %s\n" % smi_base)
            f.write("\n")

            f.write("Similar molecules\n")
            for smi in smis:
                f.write("  %s\n" % smi)

        img = Draw.MolsToGridImage([mol_base], subImgSize=(400, 200))
        img.save(os.path.join(self.gen_folder, 'noise_test/base_mol.png'))

        img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(400, 200))
        img.save(os.path.join(self.gen_folder, 'noise_test/similar_mols_generated.png'))

    def similar_gen_test(self, distances=(1, 5, 10)):
        """
        Check if novel molecules more or less similar to a lead molecule can be by adding
        a bit of randomness to an existing latent vector
        """
        print("\n# Similar molecules generator test ...")

        if not os.path.exists(os.path.join(self.gen_folder, 'similar_gen_test')):
            os.mkdir(os.path.join(self.gen_folder, 'similar_gen_test'))

        # Base molecule
        idx = np.random.choice([i for i in range(self.x_sample.shape[0])], 1, replace=False)[0]

        smi_base = "".join(
            [self.st.int_to_char[idx] for idx in
             np.argmax(np.squeeze(self.x_sample[idx:idx + 1], axis=0), axis=1)])
        smi_base = "".join([c for c in smi_base if c not in ['G', 'E', 'A']])
        latent_base = self.smi2latent_model.predict(self.x_sample[idx:idx + 1])
        mol_base = Chem.MolFromSmiles(smi_base)

        # Get 2D latent representation of all test molecules and the base molecule
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent)
        latent_2d_base = pca.transform(latent_base)

        # Get parametric equations of the circumference centered at latent_2d_base
        xc = latent_2d_base[:, 0]
        yc = latent_2d_base[:, 1]

        gen_points_dic = dict()
        gen_points_dic[0] = [(xc, yc)]
        theta = np.linspace(0, 2 * np.pi, 9)
        for r in distances:
            gen_points_dic[r] = []
            x = xc + r * np.cos(theta)
            y = yc + r * np.sin(theta)
            gen_points_dic[r] += list(zip(x, y))

        # Sample latent 2D representation
        num_samples = min(1000, latent_2d.shape[0])
        idxs = np.random.choice([i for i in range(latent_2d.shape[0])], num_samples, replace=False)
        latent_2d_sampled = latent_2d[idxs, :]
        plot_2d_distribution(latent_2d_sampled,
                             os.path.join(self.gen_folder, 'similar_gen_test/latent_2d_dist.png'),
                             points_dic=gen_points_dic)

        # Convert back generated 2D latent data to its original dimension
        latent_gen_dic = {}
        smiles_dic = {}
        mols_dic = {}
        for r, points in gen_points_dic.items():
            if r == 0:
                latent_gen_dic[r] = latent_base
                smiles_dic[r] = [smi_base]
                mols_dic[r] = [mol_base]
                continue
            latent_gen = pca.inverse_transform(points)
            smiles_gen = [latent2smiles(self.lat2states_model, self.sample_model, latent_gen[i:i+1], self.config.max_len)
                          for i in range(latent_gen.shape[0])]
            smiles_gen = list(set(smiles_gen))
            smiles_gen_valid = filter_valid_mols(smiles_gen)
            smiles_dic[r] = smiles_gen_valid
            mols_dic[r] = [Chem.MolFromSmiles(smi) for smi in smiles_gen_valid]

        with open(os.path.join(self.gen_folder, 'similar_gen_test/results.txt'), 'w') as f:
            for r, smi_list in smiles_dic.items():
                if r == 0:
                    f.write("Base molecule:\n")
                    f.write("  %s\n" % smi_list[0])
                    continue

                f.write("Distance %.2f:\n" % r)
                for smi in smi_list:
                    f.write("  %s\n" % smi)

        for r, mol_list in mols_dic.items():
            img = Draw.MolsToGridImage(mol_list, subImgSize=(400, 200))
            img.save(os.path.join(self.gen_folder, 'similar_gen_test/distance_%.2f.png' % r))


if __name__ == '__main__':
    generator = Generator()
    """
    generator.similarity_test()
    generator.interpolation_test()
    generator.noise_test()
    generator.similar_gen_test()
    """
    generator.gen_new_mols(num_mols=1000, qed_thresh=None, max_carbons=6)


