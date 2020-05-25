import configparser


class ParseConfig(object):
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        # general
        self.tf_use_gpu = config['GENERAL'].getboolean('tf_use_gpu')
        self.seed = int(config['GENERAL']['seed'])

        # data preprocessing parameters
        self.min_len = int(config['DATA_PREPROCESSING']['smiles_min_len'])
        self.max_len = int(config['DATA_PREPROCESSING']['smiles_max_len'])

        # model parameters
        self.model_name = config['MODEL']['model_name']
        self.latent_dim = int(config['MODEL']['latent_dim'])
        self.lstm_dim = int(config['MODEL']['lstm_dim'])

        # train parameters
        self.train_test_rate = float(config['TRAIN']['train_test_rate'])
        self.train_epochs = int(config['TRAIN']['train_epochs'])
        self.train_batch_size = int(config['TRAIN']['train_batch_size'])
        self.train_learning_rate = float(config['TRAIN']['train_learning_rate'])
        self.train_num_samples = int(config['TRAIN']['train_num_samples'])

        # finetune parameters
        self.ft_test_rate = float(config['FINETUNE']['ft_test_rate'])
        self.ft_epochs = int(config['FINETUNE']['ft_epochs'])
        self.ft_batch_size = int(config['FINETUNE']['ft_batch_size'])
        self.ft_learning_rate = float(config['FINETUNE']['ft_learning_rate'])
        self.ft_num_samples = int(config['FINETUNE']['ft_num_samples'])

        # callbacks parameters
        self.verbose_training = config['CALLBACKS'].getboolean('verbose_training')
        self.monitor = config['CALLBACKS']['monitor']
        self.mode = config['CALLBACKS']['mode']
        self.save_best_only = config['CALLBACKS'].getboolean('save_best_only')
        self.save_weights_only = config['CALLBACKS'].getboolean('save_weights_only')
        self.verbose = int(config['CALLBACKS']['verbose'])
        self.rlr_patience = int(config['CALLBACKS']['rlr_patience'])
        self.es_patience = int(config['CALLBACKS']['es_patience'])
