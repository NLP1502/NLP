# -*- coding: utf-8 -*-

import argparse
from utils.data import Data
from utils.data import init_parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Low Resource NER via Cross-lingual Knowledge-Transfer')
    parser = init_parser(parser)

    substring_file_prefix = 'ned'

    args = parser.parse_args()
    data = Data()
    data.read_config(args.config)
    data.read_arg(args)
    data.show_data_summary()

    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    # data.build_alphabet_substring(data.substring_dir, substring_file_prefix)
    if data.use_trans:
        data.build_translation_alphabet(data.trans_dir)
    data.fix_alphabet()
    if data.use_trans:
        data.build_translation_dict(data.trans_dir)

    data.generate_instance('train')
    data.generate_instance('dev')
    data.generate_instance('test')
    data.generate_instance_substring(substring_file_prefix)
    data.build_pretrain_emb()

    save_data_name = data.data_bin_dir
    data.save(save_data_name)
    data.show_data_summary()