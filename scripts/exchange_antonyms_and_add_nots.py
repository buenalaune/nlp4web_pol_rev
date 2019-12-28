import spacy

# Here is the accuracy for domain: 91.25 %
# Here is the accuracy for polarity: 50.24999999999999 %
# Number of correctly classified positive sentences: 1
# Number of correctly classified negative sentences: 200

with  open("../evaluation_examples.csv") as in_file:
    with open("../out.csv", "w") as out_file:
        for line in in_file:
            split = line.split(",")
            text = ",".join(split[:-2])
            domain = split[-2]
            category = split[-1]

            nlp = spacy.load('en_core_web_sm')

            tokens = nlp(line)
            tokens_new = []
            for tup in tokens:
                token = tup.text
                new = token
                synsets = tup._.wordnet.synsets()
                pos_tags = [synset.pos() for synset in synsets]
                if len(pos_tags) != 0:
                    # Replace token with its antonym (if available)
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
                # Add a not after a verb
                if tag == "VERB" or tag == "AUX":
                    tokens_new.append("not")
            text_new = tokenizer.detokenize(tokens_new).replace("\\\"", "\"").replace("''", "\"").replace("``",
                                                                                                          "\"").replace(
                "not not", "")

            out_file.write(text_new + "," + domain + "," + category)
