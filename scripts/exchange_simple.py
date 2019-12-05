import sys
sys.path.append('..')
import spacy
import pandas as pd
import util.word_pol_exchange as wpe

# Read in data
data = pd.read_csv('../evaluation_examples.csv', header=None)
data.columns = ['sentence', 'category', 'sentiment']

# Load language model
nlp = spacy.load('en_core_web_md')

# Exchange words
with open('../evaluation_examples_word_ex_simple.csv', 'w') as file:
    for row in data.itertuples():
        sentiment = 1 if row.sentiment == 3 else 0
        file.write(
            '"'
            + ' '.join(
                t for t
                in wpe.simple_exchange(
                    nlp(row.sentence),
                    sentiment,
                    ['ADJ', 'ADV'],
                    ['good', 'well'],
                    ['bad', 'bad']
                )
                if not ('"' in t)
            )
            + '",'
            + str(row.category) + ','
            + str(row.sentiment) + '\n'
        )