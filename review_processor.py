import pandas as pd
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import numpy as np


nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
df = pd.read_csv('evaluation_examples.csv', header=None) #names = ['review','domain','polarity']

# verbs which influences the polarity of sentiment significantly
verbs = ['regret', 'believe', 'dislike', 'like', 'recommend', 'waste', 'fail', 'love', 'disappoint', 'hate']

# This functions replaces the tokens in the given sentence
# 'sentence': where words shuld be replaced
# 'toBeReplaced': list of tokens to be replaced
# 'toRepace': list of tokens which replaces the tokens in the sentence
def replace_tokens(sentence, toBeReplaced, toReplace):
    new_sentence = ""
    for replaced, replacing in zip(toBeReplaced, toReplace):
        new_sentence = sentence.replace(replaced, replacing)
        sentence = new_sentence
    return sentence

number_of_reviews = len(df[0])
reviews = []
for review_id in range(number_of_reviews):
    review = ""
    review = df[0][review_id]   # single review
    nlp_review = nlp(review)
    toReplace = []              # tokens which should be replaced
    replacing = []              # tokens which replaces

    for txt in nlp_review:
        # checks if the token is verb which matters the sentiment polarity and finds its opposite verb
        if txt.pos_ == 'VERB' and txt.lemma_ in verbs:
            for syn in txt._.wordnet.synsets():
                for l in syn.lemmas():
                    if l.antonyms():
                        toReplace.append(txt.text)
                        replacing.append(l.antonyms()[0].name())

        # if the opposite of verb is not in spacy wordnet, find the similarity score with "like" and "dislike"
        # if the similarity score is closer to "like", verb is replaced with "dislike"
        # if the similarity score is closer to "dislike", verb is replaced with "like"
        if txt.pos_ == 'VERB' and txt.lemma_ in verbs and txt.text not in toReplace:
            if txt.similarity(nlp("like")) > txt.similarity(nlp("dislike")):
                toReplace.append(txt.text)
                replacing.append("dislike")
            else:
                toReplace.append(txt.text)
                replacing.append("like")

        # it tries to find the opposite of tokens which are adjectives or adverbs
        if txt.pos_ in ['ADV','ADJ']:
            for syn in txt._.wordnet.synsets():
                for l in syn.lemmas():
                    if l.antonyms():
                        toReplace.append(txt.text)
                        replacing.append(l.antonyms()[0].name())

        # if the opposite of adjectives and adverbs does not exist in wordnet, 
        # it finds the similarity score with "bad" and "bad"
        # then replaces the tokens accordingly
        if txt.pos_ in ['ADV','ADJ'] and txt.text not in toReplace:
            if txt.similarity(nlp("good")) > txt.similarity(nlp("bad")):
                toReplace.append(txt.text)
                replacing.append("bad")
            else:
                toReplace.append(txt.text)
                replacing.append("good")


    # new review after processing the old review    
    new_review = "{}".format(replace_tokens(review, toReplace, replacing))

    # Insert nots before every verb
    nlp_review = nlp(new_review)
    new_tokens = []
    for i in range(len(nlp_review)):
        if nlp_review[i].pos_ == 'VERB':
            new_tokens.append('not')
        new_tokens.append(nlp_review[i].text)
    new_review = ' '.join(new_tokens).replace('not not','')




    #adds the the list of new reviews
    reviews.append(new_review)

# replaces the review field in the dataframe
df[0] = reviews

df.to_csv('new.csv', header=None, index=False)
