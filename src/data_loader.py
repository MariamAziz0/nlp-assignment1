import pandas as pd
from tabulate import tabulate

class DataLoader:
    def __init__(self, train_file=None, test_file=None, validation_file=None, columns_to_select=None,):
        if train_file is None:
            train_file = '../dataset/sst_train.csv'
        if test_file is None:
            test_file = '../dataset/sst_test.csv'
        if validation_file is None:
            validation_file = '../dataset/sst_validation.csv'
        if columns_to_select is None:
            columns_to_select = ['sentence', 'label', 'tokens']

        self.train_file = train_file
        self.test_file = test_file
        self.validation_file = validation_file
        self.columns_to_select = columns_to_select
        self.train_df = None
        self.test_df = None
        self.validation_df = None

    def load_preprocess_data(self):
        try:
            self.train_df = pd.read_csv(self.train_file, usecols=self.columns_to_select)
            self.test_df = pd.read_csv(self.test_file, usecols=self.columns_to_select)
            self.validation_df = pd.read_csv(self.validation_file, usecols=self.columns_to_select)
            self._preprocess_data()
        except FileNotFoundError as e:
            print(f'Error while reading the data: {e}')

        return self.train_df, self.test_df, self.validation_df

    def _preprocess_data(self):
        score_mapper = lambda x: 0 if x <= 0.2 else (1 if x <= 0.4 else (2 if x <= 0.6 else (3 if x <= 0.8 else 4)))
        self.train_df['label'] = self.train_df['label'].apply(score_mapper)
        self.test_df['label'] = self.test_df['label'].apply(score_mapper)
        self.validation_df['label'] = self.validation_df['label'].apply(score_mapper)

    def get_data(self):
        return self.train_df, self.test_df, self.validation_df

    def describe_data(self, show_data=False, rows_count=2):
        dimensions_data = [
            [
                'Training Dataframe',
                self.train_df.shape[0],
                self.train_df.shape[1],
                ', '.join(self.train_df.columns),
                self.train_df['label'].value_counts()
            ],
            [
                'Testing Dataframe',
                self.test_df.shape[0],
                self.test_df.shape[1],
                ', '.join(self.test_df.columns),
                self.test_df['label'].value_counts()
            ],
            [
                'Validation Dataframe',
                self.validation_df.shape[0],
                self.validation_df.shape[1],
                ', '.join(self.validation_df.columns),
                self.validation_df['label'].value_counts()
            ],
        ]

        print(
            tabulate(
                dimensions_data,
                headers=['DataFrame', '# of Rows', '# of Columns', 'Column Names', 'Label Statistics'],
                tablefmt="grid"
            )
        )

        if show_data:
            print(self.train_df.head(rows_count))
            print(self.test_df.head(rows_count))
            print(self.validation_df.head(rows_count))