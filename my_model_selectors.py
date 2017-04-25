import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('inf')
        best_model = None

        for number in range(self.min_n_components, self.max_n_components):

            try:
                model = self.base_model(number)
                logL = model.score(self.X, self.lengths)
                transition_prob = model.transmat_
                start_prob = model.startprob_
                p = len(transition_prob) + len(start_prob)
                N = len(self.X)

                BIC = -2 * logL + p * np.log(N)

                if BIC < best_score:
                    best_model = model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float('-inf')

        keys = list(self.hwords.keys())
        other_keys = keys.remove(self.this_word)

        sum_logs = 0

        for number in range(self.min_n_components, self.max_n_components):
            model = None
            score = 0
            try:
                # This word model
                model = self.base_model(number)
                logL_word = model.score(self.X, self.lengths)

                # Other words models
                for key in other_keys:
                    X, lengths = self.hwords[key]
                    logL = model.score(X, lengths)
                    sum_logs.append(logL)

                score = logL_word - 1/(len(keys) - 1) * sum_logs
            except:
                pass
            if score > best_score:
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        max_score = float('-inf')
        split_method = KFold(min(3, len(self.sequences[0])))

        for number in range(self.min_n_components, self.max_n_components):
            set_score = []
            for cv_train, cv_test in split_method.split(self.sequences[0]):
                try:
                    X_train, lengths_train = combine_sequences(cv_train, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_train, self.sequences)

                    model = self.base_model(number)
                    model.fit(X_train, lengths_train)
                    logL = model.score(X_test, lengths_test)
                    set_score.append(logL)

                    if np.mean(set_score) > max_score:
                        best_model = model
                except:
                    pass
        return best_model
