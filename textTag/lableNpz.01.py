#coding:utf8

'''
生成每个qid对应的label,保存成json
'''
def main(file,outfile, oriNpz):
    import json
    with open(oriNpz, 'r') as f:
        dataOri = json.load(f)

    with open(file) as f:
        lines = f.readlines()

    def process(line):
        # print(line)
        qid, labels = line.replace('\n', '').split('\t')
        labels = labels.split('|')
        return qid, labels


    results = list( map(process, lines))
    # import ipdb;ipdb.set_trace()
    d = {ii:[dataOri['label2id'][jj] for jj in labels ]  for ii,labels in results}

    data = dict(d=d,label2id=dataOri['label2id'],id2label=dataOri['label2id'])
    import json
    with open(outfile,'w') as f:
        json.dump(data,f) 

if __name__=='__main__':
    import fire
    fire.Fire()
    # main("../../question_topic_train_set.txt", "labels.json")
    # main("../../question_topic_set_06_raw.txt", "labels_06_test.json")