How to run:
environment: python 2.7, torch 0.4
use "python trainCET.py" to run this demo.

Objectives: 
Guess the answer to a question from the answers of two 
questions before it and two questions after it. 
(dataset: CET-6 multiple choice questions) 

Methods: 
Use the answers of listening comprehension questions of 
CET-6 as the original dataset, and then extract every 
5-consecutive answer to establish the final dataset. 
Input data format:	A B X B D 
Output data format:	A B D B D 
ABCD are answers to questions, and X is the answer need 
to be predicted. 
Using LSTM_CRF_pytorch_demo as the baseline model. 
Here some modifications were made to the baseline. 
Firstly, adding save & load function that allows our 
model rollback to the present optimal solution when 
the performance degrades after updating. Secondly, 
Modifying the scoring function by prediction stage to 
only consider the prediction accuracy of X. 

Effect: 
The accuracy is 0.72% for the CET-6 problem in June.2018 

Defects: 
1. The scoring system in the training phase should be redesigned.
2. The gradient update method and the model need to be fine turn. 
3. Need to simplify the model to a concise formula.
4. Now the input answers are standard answers that the error rate 
have not been taken into account.
5. The dataset is small. 
6. Whether data from different sources or from too many years ago 
are valid or not still needs to be studied experimentally. 