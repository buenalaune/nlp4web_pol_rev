import nltk
from nltk.corpus import wordnet

from nltk.tokenize.treebank import TreebankWordDetokenizer

#Here is the accuracy for domain: 91.25 %
#Here is the accuracy for polarity: 62.5 %
#Number of correctly classified positive sentences: 78
#Number of correctly classified negative sentences: 172

tokenizer =TreebankWordDetokenizer()
with  open("../evaluation_examples.csv") as in_file:
    with open("../out.csv", "w") as out_file:
        for line in in_file:
            split = line.split(",")
            text = ",".join(split[:-2])
            domain= split[-2]
            category = split[-1]

            tokens = nltk.word_tokenize(text)
            tokens_new = []
            for token in tokens:
                new = token
                # Replace token with its antonym (if available)
                synsets = wordnet.synsets(token.lower())
                for synset in synsets:
                    if token != new:
                        break
                    lemmas = synset.lemmas()
                    for lemma in lemmas:
                        antonyms = lemma.antonyms()
                        if len(antonyms) != 0:
                            new = antonyms[0].name()
                            break
                if new == token:
                    # No antonym was found. Try to find an antonym for similar words
                    for synset in synsets:
                        if token != new:
                            break
                        lemmas = synset.lemmas()
                        similars = synset.similar_tos()
                        for similar in similars:
                            if token != new:
                              break
                            for lemma in similar.lemmas():
                                antonyms = lemma.antonyms()
                                if len(antonyms) != 0:
                                    new = antonyms[0].name()
                                    break

                tokens_new.append(new)
            text_new = tokenizer.detokenize(tokens_new).replace("\\\"","\"").replace("''","\"").replace("``","\"").replace("not not", "")
            out_file.write(text_new+","+domain+","+category)
