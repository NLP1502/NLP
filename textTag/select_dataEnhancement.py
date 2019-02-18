import glob
import fire
import csv
def main(**kwargs):
    for k, v in kwargs.items():
        print(k)
        print(v)
        if k == 'inpath':
            inpath = v
        if k == 'lablefile':
            lablefile = v
        if k == 'outfile':
            outfile = v
        if k == 'minSame':
            minSame = v
        if k == 'trueTitleFile':
            trueTitleFile = v

    lables = {}
    lf = open(lablefile, 'r', encoding='utf8')
    csv_reader = csv.reader(lf)
    for line in csv_reader:
        lables[line[0].strip()] = ''
    lf.close()
    # print(lables)

    fttf = open(trueTitleFile, 'r', encoding='utf8')
    trueTitle = fttf.readlines()
    findSame = dict()
    for _ in range(len(trueTitle)):
        findSame[trueTitle[_].strip().replace('/t', ' ')] = ''

    files = glob.glob(inpath + '/*.txt')
    dataset = {}
    total = 0
    repeat = 0
    bad = 0
    for file in files:
        print(file)
        if int(file.split('_')[-1].split('.')[0]) >= minSame:
            fin = open(file, 'r', encoding='utf-8')
            datas = fin.readlines()
            print('lenth: %d'%(len(datas)))
            total += len(datas)
            count = 0
            line_num = 0
            for line in datas:
                if line_num % 10000 == 0:
                    print('dealing lines: %d' % (line_num))
                line_num += 1
                title = line.split('\t', 1)[0]
                if title.strip() in findSame:
                    count += 1
                    continue
                if line.strip() not in dataset:
                    dataset[line.strip()] = ''
                else:
                    repeat += 1
            print("repeat rate: %.2f%%" % (count / len(datas) * 100))
        print('done')
    print('making id and clear lables')
    fout = open(outfile, 'w', encoding='utf-8')
    writer = csv.writer(fout)
    id = 5000000000
    totalCount = len(dataset)
    for k,v in dataset.items():
        if id % 1000 == 0:
            print('dealing id: %d/%d'%(id - 5000000000, totalCount))
        id += 1
        linecsv = (str(id) + '\t' + k.strip()).split('\t')
        if len(linecsv) == 0 or len(linecsv) < 4:
            bad += 1
            continue
        if linecsv[3][0] == '"':
            linecsv[3] = linecsv[3][1:]
        if linecsv[3][-1] == '"':
            linecsv[3] = linecsv[3][:-1]
        lableitem = linecsv[3].split(',')
        new_lable = ''
        has = 0
        for _ in lableitem:
            try:
                lables[_.strip()]
                if has == 0:
                    new_lable = _.strip()
                    has = 1
                else:
                    new_lable += '|' + _.strip()
            except:
                pass
        linecsv[3] = new_lable
        if new_lable != '':
            writer.writerow(linecsv)
        else:
            # print(lableitem)
            bad += 1
    print('total: %d, repeat: %d, bad: %d'%(total, repeat, bad))

if __name__ == '__main__':
    fire.Fire()