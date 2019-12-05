import sys
sys.path.append('..')
import spacy
import pandas as pd
import util.word_pol_exchange as wpe

# Define tags of words to be exchanged
tags = ['ADJ', 'ADV']

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
extract_df = wpe.make_extract_df(data, tags, nlp)

# Calculate word scores
score_df = wpe.make_scores(extract_df, num_words1, num_words0)

# Get words with highest and lowest scores from dataframe
pos_words = [
    str(score_df[
        score_df['tag'] == tag
    ].sort_values('score').iloc[-1]['word'])
    for tag in tags
]

neg_words = [
    str(score_df[
        score_df['tag'] == tag
    ].sort_values('score').iloc[0]['word'])
    for tag in tags
]

print(pos_words)
print(neg_words)

# Exchange words
with open('../evaluation_examples_word_ex_simple_l_h.csv', 'w') as file:
    for row in data.itertuples():
        sentiment = 1 if row.sentiment == 3 else 0
        file.write(
            '"'
            + ' '.join(
                t for t
                in wpe.simple_exchange(
                    nlp(row.sentence),
                    sentiment,
                    tags,
                    pos_words,
                    neg_words
                )
                if not ('"' in t)
            )
            + '",'
            + str(row.category) + ','
            + str(row.sentiment) + '\n'
        )