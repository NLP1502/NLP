# -*- coding: utf-8 -*-
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
import fire

# train_file='eng.train.10-16.(0.095).21943.1937.(12.454)',
def main(train_file):

    # 1. get the corpus
    # define columns
    columns = {0: 'text', 1: '', 2:'', 3: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = './eng_data_mini_onefile/'

    # retrieve corpus using column format, data folder and the names of the train, dev and test files
    corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                                  train_file=train_file,
                                                                  test_file='eng.testb',
                                                                  dev_file='eng.testa')

    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [

        WordEmbeddings('glove'),

        # comment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use flair embeddings
        # FlairEmbeddings('news-forward'),
        # FlairEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/example-ner',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150)

    # 8. plot training curves (optional)
    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
    plotter.plot_weights('resources/taggers/example-ner/weights.txt')

if __name__ == '__main__':
    fire.Fire(main)