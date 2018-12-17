# LSTM_CRF_pytorch_for_CET
Copyright 2017-YuejiaXiang, NLP Lab., Northeastern university  
Thanks to contributers: K.L. Zeng

## Introduce
This demo is a varant of LSTM_CRF_pytorch_demo in which the task it's facing is no longer NER but CET(College English Test).

## How to run:
environment: python 2.7, torch 0.4  
use "python trainCET.py" to run this demo.
 
## Objectives: 
Guess the answer of a question from the answers of two questions before it and two after.   
(dataset: CET-6 multiple choice questions) 
 
## Methods: 
- Use the listening comprehension question's answers of CET-6 as the original dataset, and then extract every 5-consecutive answers to establish the final dataset.  
Input data format: A B X B D  
Output data format: A B D B D  
ABCD are the answers of questions, and X is the answer need to be predicted.  
- Using the LSTM_CRF_pytorch_demo as baseline model.  
And then I make some modifications to the baseline model.
Firstly, I add save & load function which allows our model rollback to the present optimal solution when the performance degrades after updating.   
Secondly, I modify the scoring function by stage that only consider the prediction accuracy of X at prediction stage.
 
## Effect: 
The accuracy is 0.72 for the CET-6 problem in June 2018
 
## Defects: 
1. The scoring system in the training phase should be redesigned.
2. The gradient update method and the model need to be fine turn. 
3. Simplifying the model to a concise formula is needed.
4. Now the input answers are standard answers that do not consider the error rate of each answer.
5. The dataset is small. 
6. Whether data from different sources or from too many years ago are valid or not still needs to be studied experimentally.
 


