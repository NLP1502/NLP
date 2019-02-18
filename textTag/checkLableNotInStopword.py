import fire
import csv
def main(**kwargs):
    for k, v in kwargs.items():
        print(k)
        print(v)
        if k == 'stopwordfile':
            stopwordfile = v
        if k == 'lablefile':
            lablefile = v

    stopworddict = {}
    fstop = open(stopwordfile, 'r', encoding='utf8')
    stopdata = fstop.readlines()
    for d in stopdata:
        stopworddict[d.strip()] = ''

    lf = open(lablefile, 'r', encoding='utf8')
    csv_reader = csv.reader(lf)
    for line in csv_reader:
        try:
            stopworddict[line[0].strip()]
            print('lable in stopword:')
            print(line[0].strip())
        except:
            pass
    lf.close()
    print('finish')

if __name__ == '__main__':
    fire.Fire()