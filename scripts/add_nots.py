import nltk

from nltk.tokenize.treebank import TreebankWordDetokenizer

#Here is the accuracy for domain: 93.25 %
#Here is the accuracy for polarity: 55.25 %
#Number of correctly classified positive sentences: 20
#Number of correctly classified negative sentences: 201

tokenizer =TreebankWordDetokenizer()
with  open("../evaluation_examples.csv") as in_file:
    with open("../out.csv", "w") as out_file:
        for line in in_file:
            # Read line
            split = line.split(",")
            text = ",".join(split[:-2])
            domain= split[-2]
            category = split[-1]

            # Tokenize the tags and predict pos tags
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens, tagset="universal")
            tokens_new = []
            for tup in tags:

                tag = tup[1]
                word = tup[0]

                tokens_new.append(word)
                # Add a not after a verb
                if tag == "VERB" or tag == "AUX":
                    tokens_new.append("not")
            # Detokenize the text back and remove "not not"
            text_new = tokenizer.detokenize(tokens_new).replace("\\\"","\"").replace("''","\"").replace("``","\"").replace("not not", "")
            out_file.write(text_new+","+domain+","+category)
