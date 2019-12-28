import sys
sys.path.append('..')
import spacy
import pandas as pd
import util.word_pol_exchange as wpe
import util.classifier as classifier

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

test = '../out/evaluation_examples_word_ex_ADJ_ADV_0.csv'

# Exchange words to antonyms according to score
with open(test, 'w') as file:
    for row in data.itertuples():
        file.write(
            '"'
            + ' '.join(
                t for t
                in wpe.exchange_words(nlp(row.sentence), ['ADJ', 'ADV'], score_df, 0, nlp)
                if not ('"' in t)
            ).replace('not not', '')
            + '",'
            + str(row.category) + ','
            + str(row.sentiment) + '\n'
        )

# Evaluate
print('Evaluation of score-based exchange of words to antonym')
classifier.calculate_domain(test)
classifier.calculate_polarity(test)

test = '../out/evaluation_examples_word_ex_ADJ_ADV_0_simple.csv'

# Exchange words to good/bad or well/badly according to score
with open(test, 'w') as file:
    for row in data.itertuples():
        file.write(
            '"'
            + ' '.join(
                t for t
                in wpe.exchange_words(
                    nlp(row.sentence),
                    ['ADJ', 'ADV'],
                    score_df,
                    0,
                    nlp,
                    simple=True,
                    pos_words=['good', 'well'],
                    neg_words=['bad', 'badly']
                )
                if not ('"' in t)
            ).replace('not not', '')
            + '",'
            + str(row.category) + ','
            + str(row.sentiment) + '\n'
        )

print('''Evaluation of score-based exchange of adjectives to good/bad and
adverbs to well/badly''')
classifier.calculate_domain(test)
classifier.calculate_polarity(test)

# Get words with highest and lowest scores from dataframe
# Define tags of words to be exchanged
tags = ['ADJ', 'ADV']

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

test = '../out/evaluation_examples_word_ex_ADJ_ADV_0_l_h.csv'

# Exchange words to highest / lowest-score words according to score
with open(test, 'w') as file:
    for row in data.itertuples():
        file.write(
            '"'
            + ' '.join(
                t for t
                in wpe.exchange_words(
                    nlp(row.sentence),
                    ['ADJ', 'ADV'],
                    score_df,
                    0,
                    nlp,
                    simple=True,
                    pos_words=pos_words,
                    neg_words=neg_words
                )
                if not ('"' in t)
            ).replace('not not', '')
            + '",'
            + str(row.category) + ','
            + str(row.sentiment) + '\n'
        )

# Evaluation
print('''Evaluation of score-based exchange of adjectives and
adverbs to highest/lowest score words''')
classifier.calculate_domain(test)
classifier.calculate_polarity(test)
