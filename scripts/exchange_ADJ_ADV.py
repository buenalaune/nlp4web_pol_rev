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

# Get the total word count in positive and negative texts
num_words1 = 0
num_words0 = 0
for row in data.itertuples():
    if row.sentiment == 3:
        for token in nlp(row.sentence):
            if not token.is_punct:
                num_words1 += 1

    if row.sentiment == 4:
        for token in nlp(row.sentence):
            if not token.is_punct:
                num_words0 += 1

# Extract all adjectives into a dataframe
extract_df = wpe.make_extract_df(data, ['ADJ', 'ADV'], nlp)

# Calculate word scores
score_df = wpe.make_scores(extract_df, num_words1, num_words0)

# Exchange words
with open('../evaluation_examples_word_ex_ADJ_ADV_0.csv', 'w') as file:
    for row in data.itertuples():
        file.write(
            '"'
            + ' '.join(
                t for t
                in wpe.exchange_words(nlp(row.sentence), ['ADJ', 'ADV'], score_df, 0)
                if not ('"' in t)
            )
            + '",'
            + str(row.category) + ','
            + str(row.sentiment) + '\n'
        )