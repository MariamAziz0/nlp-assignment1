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

    def _construct_labels(self, y, num_of_classes):
        labels = np.zeros((len(y), num_of_classes), dtype=int)
        for i in range(len(y)):
            labels[i][y[i]] = 1

        return labels

    def _softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def fit(self, X_train, y_train, X_validation, y_validation, num_of_epochs=50):
        self._initialize_bigrams(X_train[:, 1])
        features = self._construct_features(X_train[:, 1])
        num_of_classes = max(y_train) + 1

        # TODO: this should be adjusted
        learning_rate = 0.001

        labels = self._construct_labels(y_train, num_of_classes)

        self.weights = np.zeros((features.shape[1], num_of_classes))
        self.biases = np.zeros((1, num_of_classes))

        for _ in range(20):
            gradient_weights = (1 / len(X_train)) * np.dot(np.transpose(self._calculate_y_hat(features) - labels), features)
            gradient_biases = (1 / len(X_train)) * np.sum(self._calculate_y_hat(features) - labels, axis=0)

            self.weights -= learning_rate * gradient_weights.T
            self.biases -= learning_rate * gradient_biases


        return self._calculate_y_hat(features), labels

    def _calculate_y_hat(self, features):
        z = np.dot(features, self.weights) + self.biases
        sigma = self._softmax(z)
        y_hat = np.zeros(z.shape, dtype=int)
        max_indices = np.argmax(sigma, axis=1)
        y_hat[np.arange(sigma.shape[0]), max_indices] = 1

        return y_hat

    def predict(self, X):
        pass
