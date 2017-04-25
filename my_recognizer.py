import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    num_items = test_set.num_items

    for word_id in range(num_items):
        X, lengths = test_set.get_item_Xlengths(word_id)
        probs = dict()
        for train_word, model in models.items():
            try:
                # print(train_word)
                # print(word_id, X, lengths)
                score = model.score(X, lengths)
                # print('Score ', score)
                probs[train_word] = score
            except:
                pass
        probabilities.append(probs)
        guess = max(probs, key=lambda i: probs[i])
        guesses.append(guess)

    return probabilities, guesses

