import csv
#配合reshapeResultCsv ，切分\分割的标签
def main(**kwargs):
    addWindowlabel = False
    for k, v in kwargs.items():
        if k == 'file':
            file = v
        if k == 'inlabelid':
            inlabelid = v
        if k == 'outfile':
            outfile = v
        if k == 'addWindowlabel':
            addWindowlabel = v

    label2id = {}
    id2label = {}

    f = open(inlabelid, 'r', encoding='utf-8')
    reader = csv.reader(f)

    firtLine = True
    for line in reader:
        print(line)
        if firtLine:
            firtLine = False
            continue
        label2id[line[0].strip()] = int(line[1].strip())-1
        id2label[int(line[1].strip())-1] = line[0].strip()
    f.close()

    if addWindowlabel:
        label2id['NUMOFLABLES1'] = len(label2id)
        label2id['NUMOFLABLES2'] = len(label2id)
        label2id['NUMOFLABLES3'] = len(label2id)
        label2id['NUMOFLABLES4'] = len(label2id)
        label2id['NUMOFLABLES5'] = len(label2id)
        id2label[len(id2label)] = 'NUMOFLABLES1'
        id2label[len(id2label)] = 'NUMOFLABLES2'
        id2label[len(id2label)] = 'NUMOFLABLES3'
        id2label[len(id2label)] = 'NUMOFLABLES4'
        id2label[len(id2label)] = 'NUMOFLABLES5'
    print(label2id)

    print('lenth of label2id: %d'%(len(label2id)))
    with open(file) as f:
        lines = f.readlines()

    def process(line):
        # print(line)
        qid,labels = line.replace('\n','').split('\t')
        labels = labels.split('|')
        return qid,labels
    results = list( map(process, lines))

    # d = {ii: [label2id[jj] for jj in labels] for ii, labels in results}
    try:
        d = {}
        for ii, labels in results:
            lr3 = []
            for jj in labels:
                lr3.append(label2id[jj])
            d[ii] = lr3
    except:
        import ipdb;
        ipdb.set_trace()


    data = dict(d=d,label2id=label2id,id2label=id2label)
    import json
    with open(outfile,'w') as f:
        json.dump(data,f) 

if __name__=='__main__':
    import fire
    fire.Fire()
    # main("../../question_topic_train_set.txt", "labels.json")
    # main("../../question_topic_set_06_raw.txt", "labels_06_test.json")