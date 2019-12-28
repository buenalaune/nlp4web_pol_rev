import sys
sys.path.append('..')
import util.classifier as classifier

test = '../out/evaluation_examples_word_ex_ADJ_ADV_0.csv'
print('Evaluation of score-based exchange of words to antonym')
classifier.calculate_domain(test)
classifier.calculate_polarity(test)

test = '../out/evaluation_examples_word_ex_ADJ_ADV_0_simple.csv'
print('''Evaluation of score-based exchange of adjectives to good/bad and
adverbs to well/badly''')
classifier.calculate_domain(test)
classifier.calculate_polarity(test)

test = '../out/evaluation_examples_word_ex_ADJ_ADV_0_l_h.csv'
print('''Evaluation of score-based exchange of adjectives and
adverbs to highest/lowest score words''')
classifier.calculate_domain(test)
classifier.calculate_polarity(test)