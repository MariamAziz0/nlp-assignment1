import numpy as np

class LogisticRegression:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.bigrams_set = set()
        self.bigram_to_index = {}
        self.weights = None
        self.biases = None

    def _get_bigrams_from_tokens(self, tokens):
        token_list = ['<S>'] + tokens.split('|') + ['<E>']
        bigrams = [(token_list[i], token_list[i + 1]) for i in range(len(token_list) - 1)]

        return bigrams

    def _initialize_bigrams(self, tokens):
        for tokens_row in tokens:
            bigrams = self._get_bigrams_from_tokens(tokens_row)
            self.bigrams_set.update(bigrams)

        self.bigram_to_index = {bigram: idx for idx, bigram in enumerate(self.bigrams_set)}

    def _construct_features(self, tokens):
        features = np.zeros((len(tokens), len(self.bigrams_set)), dtype=int)
        for i in range(len(tokens)):
            bigrams = self._get_bigrams_from_tokens(tokens[i])

            for bigram in bigrams:
                if bigram in self.bigram_to_index:
                    features[i][self.bigram_to_index[bigram]] = 1

        return features

    def _softmax(self, X):
        pass

    def fit(self, X_train, y_train, X_validation, y_validation, num_of_epochs=50):
        self._initialize_bigrams(X_train[:, 1])
        features = self._construct_features(X_train[:, 1])
        return features

    def predict(self, X):
        pass