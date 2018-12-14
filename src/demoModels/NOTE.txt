[HELP]

1.RUN
environment: python 2.7, torch 0.4
use "python LSTM_CRF_pytorch.py" to run this demo.

2.DATASET
This is a toy data just for trying the model.
sentence: o r d e r
label:    1 0 0 1 0
Sentence is a sequence of character of a word.
Label of each character dependes on whether the character is vowels or 
not. 

[NOTES]

forward_score:
gold_score:
neg_log_likelihood = forward_score - gold_score.
When training we use neg_log_likelihood as loss, when predicting we use 
_viterbi_decode method to generate prediction.

[PROBLEMS]

1.RUN ON CUDA
As the baseline (bi-LSTM CRF model which is provided by pytorch tutorial 
file) of our code do not supports cuda excutable.
We first tried "model.cuda()" statement, but this not robust,so we use:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
You can replace "cuda:0" with "cuda" and then use CUDA_VISIBLE_DEVICES=0 
to control cada visible.
After adding 'model.to(device)', all tensors defined in the class BiLSTM_CRF 
should do similar changes.As show in statement "model(precheck_sent)", 
precheck_sent is an cpu tensor, our model will automaticly change it into 
gpu tensor if run on cuda.
I don't know whether a better method can be used to deal this problem or not.

2.CRF RUN ON BATCH
Current version can only deal sentences one by one on both training and 
predicting ways.