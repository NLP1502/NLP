import json
from tqdm import tqdm
def main(inlabel2id, intrainset, inevalset, outtrainword, outtrainchar, outevalword, outevalchar, outtraintarget, outevaltarget):
    label2id = json.load(open(inlabel2id, 'r'))['d']
    fint = open(intrainset, 'r', encoding='utf-8')
    fine = open(inevalset, 'r', encoding='utf-8')
    fouttw = open(outtrainword, 'w', encoding='utf-8')
    fouttc = open(outtrainchar, 'w', encoding='utf-8')
    foutew = open(outevalword, 'w', encoding='utf-8')
    foutec = open(outevalchar, 'w', encoding='utf-8')
    fouttt = open(outtraintarget, 'w', encoding='utf-8')
    foutet = open(outevaltarget, 'w', encoding='utf-8')
    data = fint.readlines()
    for i in tqdm(range(len(data))):
        line = data[i]
        qid,tc,tw,cc,cw = line.replace('\n','').split('\t')
        outlist = (tc + ',' + cc).split(',')
        fouttc.write(' '.join(outlist) + '\n')
        outlist = (tw + ',' + cw).split(',')
        fouttw.write(' '.join(outlist) + '\n')
        # print(qid)
        # print(label2id[qid])
        labletext = ''
        for id in label2id[qid]:
            labletext += str(id) + ' '
        fouttt.write(labletext.strip() + '\n')

    data = fine.readlines()
    for i in tqdm(range(len(data))):
        line = data[i]
        qid,tc,tw,cc,cw = line.replace('\n','').split('\t')
        outlist = (tc + ',' + cc).split(',')
        foutec.write(' '.join(outlist) + '\n')
        outlist = (tw + ',' + cw).split(',')
        foutew.write(' '.join(outlist) + '\n')
        labletext = ''
        for id in label2id[qid]:
            labletext += str(id) + ' '
        foutet.write(labletext.strip() + '\n')

if __name__=='__main__':
    import fire
    fire.Fire()