import numpy as np

class LogisticRegression:
    def __init__(self, random_state=42):
        self.bigrams_set = set()
        self.bigram_to_index = {}
        self.weights = None
        self.biases = None
        np.random.seed(random_state)

    def fit(self, X_train, y_train, X_validation, y_validation, num_of_epochs=20, batch_size=500, learning_rate=0.5, rate_decay=0.5, patience=5):
        self._initialize_bigrams(X_train[:, 1])
        features = self._construct_features(X_train[:, 1])
        features_validation = self._construct_features(X_validation[:, 1])
        num_of_classes = max(y_train) + 1
        best_validation_loss = np.inf
        current_patience_counter = 0

        labels = self._construct_labels(y_train, num_of_classes)

        self.weights = np.zeros((features.shape[1], num_of_classes))
        self.biases = np.zeros((1, num_of_classes))

        for i in range(num_of_epochs):
            train_loss = self._calculate_cross_entropy_loss(features, y_train)
            validation_loss = self._calculate_cross_entropy_loss(features_validation, y_validation)

            if i % 10 == 0:
                print(f'Epoch {i}: Train loss = {train_loss}, Validation loss = {validation_loss}')

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                current_patience_counter = 0
            else:
                current_patience_counter += 1

                if current_patience_counter >= patience:
                    learning_rate = learning_rate * rate_decay
                    current_patience_counter = 0
                    print(f'Epoch {i}: Reducing learning rate to {learning_rate}')

            indices = np.random.permutation(len(X_train))

            for j in range(len(X_train) // batch_size + 1):

                batch_indices = indices[j * batch_size:(j + 1) * batch_size]

                if len(batch_indices) == 0:
                    break

                y_hat = self._calculate_y_hat(features[batch_indices])

                gradient_weights = (1 / len(batch_indices)) * np.dot(np.transpose(
                    y_hat - labels[batch_indices]
                ), features[batch_indices])

                gradient_biases = (1 / len(batch_indices)) * np.sum(y_hat - labels[batch_indices], axis=0)

                self.weights -= learning_rate * gradient_weights.T
                self.biases -= learning_rate * gradient_biases

        return

    def predict(self, X):
        features = self._construct_features(X[:, 1])
        return np.argmax(self._calculate_y_hat(features), axis=1)

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
        features = np.zeros((len(tokens), len(self.bigrams_set)), dtype=np.int8)
        for i in range(len(tokens)):
            bigrams = self._get_bigrams_from_tokens(tokens[i])

            for bigram in bigrams:
                if bigram in self.bigram_to_index:
                    features[i][self.bigram_to_index[bigram]] = 1

        return features

    def construct_features(self, tokens):
        features = np.zeros((len(tokens), len(self.bigrams_set)), dtype=np.int8)
        for i in range(len(tokens)):
            bigrams = self._get_bigrams_from_tokens(tokens[i])

            for bigram in bigrams:
                if bigram in self.bigram_to_index:
                    features[i][self.bigram_to_index[bigram]] = 1

        return features

    def _construct_labels(self, y, num_of_classes):
        labels = np.zeros((len(y), num_of_classes), dtype=np.int8)
        for i in range(len(y)):
            labels[i][y[i]] = 1

        return labels

    def _softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _calculate_y_hat(self, features):
        z = np.dot(features, self.weights) + self.biases
        sigma = self._softmax(z)
        y_hat = np.zeros(z.shape, dtype=int)
        max_indices = np.argmax(sigma, axis=1)
        y_hat[np.arange(sigma.shape[0]), max_indices] = 1

        # return y_hat
        return sigma

    def _calculate_cross_entropy_loss(self, features, y):
        z = np.dot(features, self.weights) + self.biases
        sigma = self._softmax(z)

        loss = 0
        for i in range(len(y)):
            loss -= np.log(sigma[i][y[i]] / sum(sigma[i]))

        return loss / features.shape[0]