import csv
def main(infile="./task6data2_new/train_data.csv", questionfile="./task6data2.txt"):
    with open(infile, 'r', encoding='utf8') as fin, open(questionfile, 'w', encoding='utf8') as fout:
        reader = csv.reader(fin)
        data = list(reader)
        for _ in data:
            fout.write(_[1] + '\n')
if __name__ == "__main__":
    main()
    # fire.Fire()
