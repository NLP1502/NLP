import torch as t
import numpy as np
import json
import ipdb
import pickle
savepath = 'tryPick'
print(savepath)
# pickle.dump({'r':label_right, 'f':label_false}, savepath)
label_right = 'a'
f = open(savepath, 'wb')
pickle.dump(label_right, f)
# with open('./labels_official_dev_set.json') as f:
#     labels_ = json.load(f)
#     # labels_['d']
#     ipdb.set_trace()
# l = [[1,2.0],[3.0,2.4]]
# n = np.array(l)
# r = t.from_numpy(n).float()
# print(r)

# index2qid = np.load('question_context_official_set.npz')['index2qid'].item()
#
#