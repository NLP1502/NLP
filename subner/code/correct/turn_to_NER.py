#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import xml.dom.minidom
import sys

def turn_to_NER(input_name, output_name): 
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(input_name)
    collection = DOMTree.documentElement
    
    docs = collection.getElementsByTagName("DOC")
    i = 0
    results = list()
    for doc in docs:
        i += 1
        if i % 100 == 0:
            print("i=", i)
        text = doc.getElementsByTagName('TEXT')[0]
        text = text.childNodes[0].data[1:-1]
    #    print(text)
    #    correction = doc.getElementsByTagName('CORRECTION')[0]
    #    correction = correction.childNodes[0].data[1:-1]
        errors = doc.getElementsByTagName('ERROR')
        error_list = list()    
        for error in errors:
            start = error.getAttribute("start_off")
            end = error.getAttribute("end_off")
            error_type = error.getAttribute("type")
            error_list.append((start, end, error_type))
        j = 0
        if j < len(error_list):
            start = int(error_list[j][0]) - 1
            end = int(error_list[j][1]) - 1
            error_type = error_list[j][2]
        else:
            start = -1
            end = -1
            error_type = None
    #    print(start,end,error_type)
        k = 0
        while k < len(text):
            if k == start:
                result = text[k] + "\tB-" + error_type + '\n'
    #            print(result)
                results.append(result)
                k += 1
                while k <= end:
                    result = text[k] + "\tI-" + error_type + '\n'
    #                print(result)
                    results.append(result)
                    k += 1
                j += 1
                if j < len(error_list):
                    start = int(error_list[j][0]) - 1
                    end = int(error_list[j][1]) - 1
                    error_type = error_list[j][2]
                else:
                    start = -1
                    end = -1
                    error_type = None
            else:
                result = text[k] + "\tO" + '\n'
    #            print(result)
                results.append(result)
                k += 1
    #    print("---------------------------------")
        result = '\n'
        results.append(result)
    
    with open(output_name, "w", encoding="utf8") as fout:
        w = 0       
        for r in results:
            w += 1
            if w % 10000 == 0:
                print("w:", w)
            fout.write(r)
        
if __name__ == "__main__":
    if len(sys.argv) == 3:
        turn_to_NER(sys.argv[1], sys.argv[2])
    else:
        input_name = "data/cor/NSMT.70w.labeled.xmlformat"
        output_name = "data/cor/NSMT.70w.labeled.NERformat"
        print("Start the default path:", input_name, output_name)
        turn_to_NER(input_name, output_name)


      
