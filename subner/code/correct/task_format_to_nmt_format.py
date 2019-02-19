input_name = 'data/cor/NLPCC2018_GEC.train'
output_src_name = 'data/cor/NLPCC2018_GEC.train.src'
output_tgt_name = 'data/cor/NLPCC2018_GEC.train.tgt'
# input_name = '../data/cor/NLPCC2018_GEC.train'
# output_src_name = '../data/cor/NLPCC2018_GEC.train.src'
# output_tgt_name = '../data/cor/NLPCC2018_GEC.train.tgt'

with open(input_name, "r", encoding="utf8") as fin, \
        open(output_src_name, "w", encoding="utf8") as fout_src, \
        open(output_tgt_name, "w", encoding="utf8") as fout_tgt:
# with open(input_name, "r") as fin, \
#         open(output_src_name, "w") as fout_src, \
#         open(output_tgt_name, "w") as fout_tgt:
    datas = fin.readlines()
    for data in datas:
        data_split = data.strip().split('\t')
        ref_num = int(data_split[1])
        if len(data_split) != 3 + ref_num:
            print(data_split)
            continue
        try:
            for i in range(ref_num):
                len_r = len(data_split[2])*1.0 / len(data_split[3+i])
                if len_r > 2 or len_r < 0.5:
                    print(data_split[2] + '\t\t' + data_split[3+i])
                    continue
                fout_src.write(data_split[2] + '\n')
                fout_tgt.write(data_split[3+i] + '\n')
        except:
            print(data)
