import numpy as np

class NaiveBayes:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.classes = sorted(train_df['label'].unique().tolist())
        self.vocab = set()
        self.big_doc = {c: {} for c in self.classes}
        self.log_prior = [None for i in range(len(self.classes))]
        self.log_likelihood = dict()

    def train(self):
        self._preprocess_vocab()
        self._compute_big_doc()
        n_doc = self.train_df.shape[0]

        finished_count = 0
        for c in self.classes:
            n_c = self.train_df.loc[self.train_df.label == c].shape[0]
            self.log_prior[c] = np.log(n_c / n_doc)

            word_occurrences_in_c = {}
            total_word_occurrences = 0
            for word_type in self.vocab:
                word_occurrences_in_c[word_type] = self.big_doc[c][word_type] if word_type in self.big_doc[c] else 0
                total_word_occurrences += word_occurrences_in_c[word_type]

            for word_type in self.vocab:
                self.log_likelihood[(word_type, c)] = np.log(
                    (word_occurrences_in_c[word_type] + 1) / (total_word_occurrences + len(self.vocab))
                )
            finished_count += n_c
            print(f'Training model .. {(finished_count / n_doc) * 100:.2f}% completed.')

    def predict(self, test_doc):
        test_doc = test_doc.split(' ')
        class_score = [0 for _ in self.classes]
        for c in self.classes:
            class_score[c] = self.log_prior[c]
            for word_token in test_doc:
                if word_token in self.vocab:
                    class_score[c] += self.log_likelihood[(word_token, c)]
        return np.argmax(class_score)

    def predict_all(self):
        y_hat = [self.predict(test_doc) for test_doc in self.test_df['sentence']]
        predicted_correctly = 0
        for i in range(len(y_hat)):
            predicted_correctly += 1 if y_hat[i] == self.test_df.label[i] else 0
        print(f'Model accuracy: {(predicted_correctly / len(self.test_df)) * 100:.2f}%')
        return y_hat

    def _preprocess_vocab(self):
        for tokens in self.train_df['tokens']:
            self.vocab = self.vocab.union(set(tokens.split('|')))
        print(f'Finished preprocessing corpus vocabulary .. found {len(self.vocab)} word type.')

    def _compute_big_doc(self):
        finished_count = 0
        for c in self.classes:
            c_docs = self.train_df.loc[self.train_df.label == c]
            for c_doc in c_docs['tokens']:
                for c_doc_token in c_doc.split('|'):
                    if c_doc_token not in self.big_doc[c]:
                        self.big_doc[c][c_doc_token] = 0
                    self.big_doc[c][c_doc_token] += 1
            finished_count += c_docs.shape[0]
            print(f'Computing big doc .. {(finished_count / self.train_df.shape[0]) * 100:.2f}% completed')

