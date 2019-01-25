import re

errors = []
data_file_name = "2016_CGED_TrainingSet.txt"
# data_file_name = "train.release.xml"
tagt_file_name = "label_result.txt"
out_file_name = "compare.txt"


with open(tagt_file_name, 'r', encoding='utf-8') as in_file:
	error = []
	for line in in_file.readlines():
		if len(line) <= 1:
			errors.append(error)
			error = []
		else:
			tmp = line.strip().split(' ')
			error.append([tmp[1], tmp[2], tmp[0]])
with open(out_file_name, 'w', encoding='utf-8') as out:
	with open(data_file_name, 'r', encoding='utf-8') as in_file:
		data = in_file.read()
		docs = re.findall(re.compile(r'<DOC>((.*[\n])+?)</DOC>'), data)
		k = 0
		print(len(docs), len(errors))
		for doc in docs:
			wrong = re.findall(re.compile(r'<TEXT id="\w+">[\n](.*)[\n]</TEXT>'), doc[0])
			correct = re.findall(re.compile(r'<CORRECTION>[\n](.*)[\n]</CORRECTION>'), doc[0])
			infos = re.findall(re.compile(r'<ERROR start_off="([0-9]+)" end_off="([0-9]+)" type="([SRWM])">'), doc[0])
			flag = False
			for info, error in zip(infos, errors[k]):
				if info != tuple(error):
					flag = True
			if flag:
				out.write(wrong[0] + '\n')
				out.write(correct[0] + '\n')
				for info, error in zip(infos, errors[k]):
					out.write(str(info) + str(error) + '\n')
				out.write('\n')
			k += 1

