import spacy
import pandas as pd
import math

# Function to extract all words which are tagged with one of the tags
# given in the tags list
# Inputs:
# words: a spacy document
# tags: a list of string tags
# returns a list of words and the corresponding list of tags in the same
# order
def extract_words(words, tags):
    return (
        [
            token.text.lower()
            for token in words
            if ((token.pos_ in tags) and not token.is_punct)
        ],
        [
            token.pos_
            for token in words
            if ((token.pos_ in tags) and not token.is_punct)]
    )

# Tokenize each sentence and extract adjectives and adverbs
def make_extract_df(data, tags, model):
    # Create empty lists to store words, tags and polarity values
    word_list = []
    tag_list = []
    pol_list = []
    # Iterate over all rows in the data
    for row in data.itertuples():
        # Extract the words of interest
        ex_words, ex_tags = extract_words(model(row.sentence), tags)
        # Extend the lists for words, tags and polarity
        word_list.extend(ex_words)
        tag_list.extend(ex_tags)
        pol_list.extend([
            1 if row.sentiment == 3 else 0
            for i in range(len(ex_words))
        ])

    return pd.DataFrame({
        'word' : word_list,
        'tag' : tag_list,
        'polarity' : pol_list
    })

# Function which takes a dataframe with columns 'word', 'tag', 'polarity'
# and returns a new dataframe with one row for each word with columns
# 'word', 'tag' and 'score'
# s = log(((f1+1)/num_tokens1)/((f0+1)/num_tokens0)) (log base 2)
# score is positive when word appears more often in positive than
# in negative texts
# returns a dict with the words as keys and the scores as values
def make_scores(raw_df, num_tokens1, num_tokens0):
    # Create empty df to store scores
    score_df = pd.DataFrame(
        {
            'word': [],
            'tag': [],
            'score': []
        }
    )

    # Get the set of word-tag tuples to generate scores for
    words_to_score = set(
        [(row.word, row.tag) for row in raw_df.itertuples()]
    )

    # Iterate over the set of word-tag tuples to calculate a score
    # for each and append to score_df
    for word, tag in words_to_score:
        # Calculate word frequency in positive and negative text
        f1 = len(
            raw_df[
                (raw_df['word'] == word)
                & (raw_df['tag'] == tag)
                & (raw_df['polarity'] == 1)

                ]
        )

        f0 = len(
            raw_df[
                (raw_df['word'] == word)
                & (raw_df['tag'] == tag)
                & (raw_df['polarity'] == 0)

                ]
        )
        word_score = math.log(
            (
                    ((f1 + 1) / num_tokens1)
                    / ((f0 + 1) / num_tokens0)
            ),
            2
        )
        score_df = score_df.append({
            'word': word,
            'tag': tag,
            'score': word_score
        }, ignore_index=True)

    return score_df


# Function which takes a text and a df with word scores
# and exchanges all words of a certain tag
# and which are above / below a certain score threshold with an
# 'opposing' word of proper score and tag
def exchange_words(text, tags, score_df, thresh, simple=False):
    # Empty list to store new (exchanged) token sequence
    new_text = []

    simple_exchange_dict = {
        'neg' : {
            'ADJ' : 'bad',
            'ADV' : 'badly'
        },
        'pos' : {
            'ADJ' : 'good',
            'ADV' : 'well'
        }
    }

    # Iterate over input text
    for token in text:
        # Check if tag of current token is to be exchanged:
        if (token.pos_ in tags) and not token.is_punct:

            try:
                # If yes, check if score is above or below threshold
                score = float(score_df.loc[
                      (score_df['word'] == token.text.lower())
                      & (score_df['tag'] == token.pos_)
                ]['score'])

            # handle exception when combination of word and tag cannot be
            # found in score_df
            except TypeError:
                score = 0

            if abs(score) > thresh:

                if not simple:
                    # If yes, append token with same tag of 'opposite' score
                    # to new_text
                    # Make new dataframe of all exchange candidates
                    ex_df = score_df.loc[
                        score_df['tag'] == token.pos_
                        ].copy()
                    # Add new column to ex_df with values abs(((-1)*score)-ex_df.loc[<row>]['score'])
                    ex_df['score_comp'] = [
                        abs(((-1) * score) - row.score) for row in ex_df.itertuples()
                    ]

                    # Get the smallest value in column 'score_comp' and exchange
                    # the original word for this word
                    new_text.append(
                        ex_df.sort_values('score_comp').iloc[0]['word']
                    )

                else:
                    # If score is positive, append negative word and
                    # vice versa

                    if score > 0:
                        new_text.append(
                            simple_exchange_dict['neg'][token.pos_]
                        )

                    else:
                        new_text.append(
                            simple_exchange_dict['pos'][token.pos_]
                        )

        else:
            # If token is not to be exchanged, append original token
            new_text.append(token.text)

    return new_text

# Function which exchanges every word of a certain sentiment and tag
# with a fixed word of opposite sentiment and the same tag
# Inputs
# text: The text of which the words are to be exchanged as a
# spacy Document object
# sentiment: The sentiment of the text (1 or 0)
# tags: The POS-tags of the words to be exchanged
# pos_words: A list of words which are to be inserted into a negative text.
# Needs to be of same length and order as tags list
# neg_words: A list of words which are to be inserted into a positive text.
# Needs to be of same length and order as tags list
def simple_exchange(
        text, sentiment, tags, pos_words, neg_words, remove_not=False):
    # make dictionary of words to be exchanged
    exchange_dict = {
        0 : pos_words,
        1: neg_words
    }

    # # Empty list to store new (exchanged) token sequence
    new_text = []

    # Iterate over all tokens in the text
    for token in text:
        # Check if token has tag which qualifies it for exchange
        if token.pos_ in tags:
            # Exchange to given exchange token
            new_text.append(
                exchange_dict[sentiment][tags.index(token.pos_)]
            )

        # Check if token is 'not' and should be removed
        elif (token.text == 'not') and remove_not:
            # if yes, do nothing
            pass

        # If none of the above, append the token as it is to the new text
        else:
            new_text.append(token.text)

    return new_text