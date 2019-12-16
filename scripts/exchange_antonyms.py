import nltk
from nltk.corpus import wordnet

from nltk.tokenize.treebank import TreebankWordDetokenizer




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
            ignore_next = False
            for token in tokens:
                new = token
                synsets = wordnet.synsets(token.lower())
                for synset in synsets:
                    if token != new:
                        break
                    lemmas = synset.lemmas()
                    for lemma in lemmas:
                        antonyms = lemma.antonyms()
                        if len(antonyms) != 0:
                            new = antonyms[0].name()
                            print(token, new)
                            break
                if new == token:
                    # Not changed
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
                                    print(token, new)
                                    break

                tokens_new.append(new)
            text_new = tokenizer.detokenize(tokens_new).replace("\\\"","\"").replace("''","\"").replace("``","\"").replace("not not", "")



            out_file.write(text_new+","+domain+","+category)
