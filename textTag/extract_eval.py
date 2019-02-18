import fire

def main(**kwargs):
    for k, v in kwargs.items():
        if k == 'infile':
            infile = v
        if k == 'out_train_set':
            out_train_set = v
        if k == 'out_eval_set':
            out_eval_set = v
        # true_data_num has no use
        if k == 'true_data_num':
            true_data_num = v
        if k == 'extract_ratio':
            extract_ratio = v
    print(extract_ratio)
    fin = open(infile, 'r')
    ftrain = open(out_train_set, 'w')
    feval = open(out_eval_set, 'w')
    file = fin.readlines()
    import random
    for _ in range(len(file)):
        if random.random() < extract_ratio:
            # print(_)
            feval.write(file[_])
        else:
            ftrain.write(file[_])



if __name__=="__main__":
    fire.Fire()