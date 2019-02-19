# -*- coding: utf-8 -*-

import argparse
from utils.data import Data
from utils.data import init_parser
from utils.functions import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Low Resource NER via Cross-lingual Knowledge-Transfer')
    parser = init_parser(parser)
    additional_suffix = '.patched_pos'

    args = parser.parse_args()
    data = Data()
    data.load(args.data_bin_dir)

    print('the pos is the first feature, rank at 0')
    if not data.feature_emb_dirs[0]:
        print('init data.feature_emb_dirs[0] is none, init is as: data/ned_pos.vector30')
        data.feature_emb_dirs[-1] = 'data/ned_pos.vector30'
    print("Load pretrained feature %s embedding:, norm: %s, dir: %s" % (
        data.feature_name[0], data.norm_feature_embs[0], data.feature_emb_dirs[0]))
    data.pretrain_feature_embeddings[0], data.feature_emb_dims[0] = build_pretrain_embedding(
        data.feature_emb_dirs[0], data.feature_alphabets[0], data.feature_emb_dims[0],
        data.norm_feature_embs[0])
    data.pretrain_pos_embedding = data.pretrain_feature_embeddings[0]
    save_data_name = data.data_bin_dir + additional_suffix
    data.save(save_data_name)