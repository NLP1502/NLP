# coding:utf8
import time
import warnings

tfmt = '%m%d_%H%M%S'


class Config(object):
    validation_num = 50000
    question_train_set_items = 1005248
    ps_outfile = 'ps_default.csv'
    loss = 'multilabelloss'
    inpath = ''
    single_model_score_file = ''
    model = 'CNNText'
    title_dim = 100
    content_dim = 200
    num_classes = 25551
    embedding_dim = 256  # embedding大小
    linear_hidden_size = 2000
    kmax_pooling = 2  # k
    hidden_size = 256  # LSTM hidden size
    num_layers = 2  # LSTM layers
    inception_dim = 512

    # vocab_size = 11973 # num of chars
    vocab_size = 153721  # num of words
    kernel_size = 3
    kernel_sizes = [2, 3, 4]
    title_seq_len = 50
    content_seq_len = 250
    type_ = 'word'  # word 和char
    all = False

    embedding_path = '../char_embedding_enhanc3.npz'  # Embedding
    train_data_path = '../train_enhance3.npz'  # train
    labels_path = '../labels_enhance3.json'  # labels
    test_data_path = '../test_enhance3.npz'  # test
    result_path = 'csv/' + time.strftime(tfmt) + '.csv'
    shuffle = True
    num_workers = 4
    pin_memory = True
    batch_size = 128

    env = time.strftime(tfmt)
    plot_every = 10

    #ensemble
    ensemble_outfile = 'ensemble_default.csv'
    do_ensemble_same_name_models = True
    do_each_two_ensemble = False

    max_epoch = 100
    lr = 5e-3
    lr2 = 1e-3
    min_lr = 1e-5
    lr_decay = 0.99
    weight_decay = 0
    weight = 1
    decay_every = 1000

    test_model_path = ''
    test_result_path = ''

    model_path = None
    optimizer_path = 'optimizer.pth'

    debug_file = '/tmp/debug2'
    debug = False

    gpu1 = False
    floyd = False
    zhuge = False

    model_names = ['MultiCNNTextBNDeep', 'CNNText_inception', 'RCNN', 'LSTMText', 'CNNText_inception']
    model_paths = ['checkpoints/MultiCNNTextBNDeep_0.37125473788', 'checkpoints/CNNText_tmp_0.380390420742',
                   'checkpoints/RCNN_word_0.373609030286', 'checkpoints/LSTMText_word_0.381833388089',
                   'checkpoints/CNNText_tmp_0.376364647145']  # ,'checkpoints/CNNText_tmp_0.402429167301']
    static = False
    val = False
    save_top_result = True
    save_top_num = 15
    visible_top_num = 10
    save_test_result = False

    fold = 1
    augument = True

    ###stack
    model_num = 7
    data_root = ""
    labels_file = "../labels_enhance3.json"
    val = "../val_enhance3.npz"


def parse(self, kwargs, print_=False):

    # import ipdb;ipdb.set_trace()
    print('*******in parse')
    for k, v in kwargs.items():
        if not hasattr(self, k):
            raise Exception("opt has not attribute <%s>" % k)
        setattr(self, k, v)
        print(k)
        print (v)
    print('*******out parse')

    if self.gpu1:
        self.train_data_path = '../train_enhance3.npz'
        self.test_data_path = '../%s.npz' % ('val_enhance3' if self.val else 'test_enhance3')
        self.labels_path = '../labels_enhance3.json'
        self.embedding_path = self.embedding_path.replace('', '')

    if self.floyd:
        self.train_data_path = '../train_enhance3.npz'
        self.test_data_path = '../%s.npz' % ('val_enhance3' if self.val else 'test_enhance3')
        self.labels_path = '../labels_enhance3.json'
        self.embedding_path = '../char_embedding_enhance3.npz'
    if self.zhuge:
        self.train_data_path = '../train_enhance3.npz'
        self.test_data_path = '../%s.npz' % ('val_enhance3' if self.val else 'test_enhance3')
        self.labels_path = '../labels_enhance3.json'
        self.embedding_path = '../char_embedding_enhance3.npz'

    if self.type_ == 'word':
        self.vocab_size = 153721  # num of words
        self.title_seq_len = 30
        self.content_seq_len = 120
        self.embedding_path = self.embedding_path.replace('char', 'word') if self.embedding_path is not None else None

    if self.type_ == 'char':
        self.vocab_size = 7377  # num of words
        self.title_seq_len = 50
        self.content_seq_len = 250

    if self.model_path:
        self.embedding_path = None

    if print_:
        print('user config:')
        print('#################################')
        for k in dir(self):
            if not k.startswith('_') and k != 'parse' and k != 'state_dict':
                print(k, getattr(self, k))
        print('#################################')
    return self


def state_dict(self):
    return {k: getattr(self, k) for k in dir(self) if not k.startswith('_') and k != 'parse' and k != 'state_dict'}


Config.parse = parse
Config.state_dict = state_dict
opt = Config()
