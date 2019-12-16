import nltk

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
            tags = nltk.pos_tag(tokens, tagset="universal")
            print(tags)
            tokens_new = []
            for tup in tags:

                tag = tup[1]
                word = tup[0]

                tokens_new.append(word)
                if tag == "VERB" or tag == "AUX":
                    tokens_new.append("not")
            text_new = tokenizer.detokenize(tokens_new).replace("\\\"","\"").replace("''","\"").replace("``","\"").replace("not not", "")



            out_file.write(text_new+","+domain+","+category)
