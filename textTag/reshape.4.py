import fire

fin = open('question_set_06.txt', 'r')
ftrain = open('question_train_set_06.txt', 'w')
feval = open('question_eval_set_06.txt', 'w')
file = fin.readlines()
import random
for _ in file:
    if random.randint(0, 100) < 10:
        feval.write(_)
    else:
        ftrain.write(_)



if __name__=="__main__":
    fire.Fire()