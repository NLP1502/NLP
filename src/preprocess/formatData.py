# -*- coding: utf-8 -*-
import glob
###
# When adding a new type, you should update the
#   determineOriginalDataType function and then
#   create a new function named format_TYPE. TYPE
#   is the name of this new type which should be
#   consistent with the name returned by function
#   determineOriginalDataType.
#
# The regular expression of the existing types of
#   data are shown as follow. \T means Tag. \C
#   means a Chinese word
# type1: \C[\s]\T\n
# type2: \C\d[\s]\T\n
# type3: [\C|\s]+[\s]\T\n
# type4: \C\d[\s]\T[\C|\s]+\n
#
# Args:
#   lines: original data
# Returns:
#   a list of [word, tag]
#
###
def format_type1(lines):
    out = []
    for line in lines:
        try:
            sent = line.decode('utf-8')
        except:
            try:
                sent = line.decode('ANSI')
            except:
                try:
                    sent = line.decode('gbk')
                except:
                    print('format_type1: can\'t decode %s' % (line))
                    continue
        seg = sent.strip().split()
        if len(seg) == 0:
            out.append('\n')
        else:
            out.append(seg[0] + '\t' + seg[1] + '\n')
    return out

def format_type2(lines):
    out = []
    for line in lines:
        try:
            sent = line.decode('utf-8')
        except:
            try:
                sent = line.decode('gbk')
            except:
                print('format_type2: can\'t decode %s'%(line))
                continue
        seg = sent.strip().split()
        if len(seg) == 0:
            out.append('\n')
        else:
            out.append(seg[0][:-1] + '\t' + seg[1] + '\n')
    return out

def format_type3(lines):
    out = []
    for line in lines:
        try:
            sent = line.decode('utf-8')
        except:
            try:
                sent = line.decode('gbk')
            except:
                print('format_type3: can\'t decode %s'%(line))
                continue
        seg = sent.strip().split()
        if len(seg) == 0:
            out.append('\n')
        else:
            words = ''
            for s in seg[:-1]:
                words += s + ' '
            for w in words.strip():
                out.append(w + '\t' + seg[-1] + '\n')
    return out

def format_type4(lines):
    out = []
    for line in lines:
        try:
            sent = line.decode('utf-8')
        except:
            try:
                sent = line.decode('gbk')
            except:
                print('format_type4: can\'t decode %s'%(line))
                continue
        seg = sent.strip().split()
        if len(seg) == 0:
            out.append('\n')
        else:
            out.append(seg[0][:-1] + '\t' + seg[1] + '\n')
    return out

def formatAndSaveAs(path, type, outFilePath = 'NULL',
        splitSentencesWithPunctuation = True, debug=True,
        tagFormat='original'):
    """format data and save output at outFilePath

    Use format_TYPE functions to transform original
    data to a list of [word, tag]. Then do some
    additional operations.

    Args:
        path: data file path
        type: the type of data
        outFilePath: where to save the output
        splitSentencesWithPunctuation: whether to split
            sentences with punctuation or not
        tagFormat： format of tag
        debug: print debug information
    Returns:
        no returns
    """
    if debug:
        print('dealing: ' + path)
    with open(path, 'rb') as fin:
        if outFilePath == 'NULL':
            outFilePath = path + '.format'
        with open(outFilePath, 'w', encoding='utf-8') as fout:
            lines = fin.readlines()
            out = eval('format_' + type + '(lines)')
            for _ in out:
                if tagFormat == 'original':
                    fout.write(_)
                elif tagFormat == 'noBIO':
                    fout.write(_)
                    print("Warning: formatAndSaveAs: not implement 'noBIO' model of tagFormat.")
                if splitSentencesWithPunctuation:
                    if len(_) > 0 and _[0] in ['。','?','!']:
                        fout.write('\n')

def determineOriginalDataType(path, checkNum = 5, readfileModel = 'buf10000', debug=False):
    """determine which type the original data is.

    use difLenOfLine、difLenOfSeg1、tagAtTail e.t.
    features to determine the type.

    Args:
        path: data file path
        checkNum: the number of sentences which are at
            the beginning of data file that will be used
            to determine the type of original data
        readfileModel: how to read file
        debug: print debug information
    Returns:
        String: the type
    """
    if debug:
        print('dealing: ' + path)
    with open(path, 'rb') as f:
        if readfileModel == 'buf10000':
            lines = f.readlines(10000)
        elif readfileModel == 'buf1000000':
            lines = f.readlines(1000000)
        elif readfileModel == 'wholeFile':
            lines = f.readlines()
        else:
            raise Exception('Unexpected readfileModel %s' % readfileModel)
        buf = []
        for line in lines:
            try:
                sent = line.decode('utf-8')
            except:
                try:
                    sent = line.decode('gbk')
                except:
                    print('determineOriginalDataType: can\'t decode %s' % (line))
                    continue
            if debug:
                print(sent.strip())
            seg = sent.strip().split()
            if len(seg) == 0:
                continue
            if seg[-1][0] in ['B', 'I', 'N', 'I', 'O']:
                atTail = True
            else:
                atTail = False
            if len(buf) < checkNum:
                buf.append([len(seg), len(seg[0]), atTail])
            else:
                break
        difLenOfLine = False
        difLenOfSeg1 = False
        tagAtTail = True
        for i in range(1,len(buf)):
            if debug:
                print(buf[i])
            if buf[i][0] != buf[i-1][0]:
                difLenOfLine = True
            if buf[i][1] != buf[i-1][1]:
                difLenOfSeg1 = True
            if buf[i][2] == False:
                tagAtTail = False
        if debug:
            print(difLenOfLine)
            print(tagAtTail)
        if buf[0][0] == 2 and buf[0][1] == 1 and not difLenOfLine and tagAtTail:
            return 'type1'
        elif buf[0][0] == 2 and buf[0][1] == 2 and not difLenOfLine and tagAtTail:
            return 'type2'
        elif difLenOfSeg1 and tagAtTail:
            return 'type3'
        elif buf[0][1] == 2 and not tagAtTail:
            return 'type4'
        else:
            return 'other type'

if __name__=='__main__':
    pathFiles = '../../data/'
    files = glob.iglob(pathFiles + '/**', recursive=True)
    suffixList = ['ner','test','train','dev','txt']
    for file in files:
        for suffix in suffixList:
            if file.endswith(suffix):
                type = determineOriginalDataType(file, checkNum=5, debug=False)
                formatAndSaveAs(file, type, tagFormat='original')
                break

