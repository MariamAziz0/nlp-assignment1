import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, y, y_hat):
        self.y = y
        self.y_hat = y_hat
        self.confusion_matrix = None
        self.separate_confusion_matrix = None
        self.macro_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        self.micro_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        self.metrics_per_class = {}

        self._generate_confusion_matrices()

    def _generate_confusion_matrices(self):
        classes_count = len(set(self.y))
        self.confusion_matrix = np.zeros((classes_count, classes_count))        # cols are real labels
        self.separate_confusion_matrix = np.zeros((classes_count, 2, 2))

        for i in range(len(self.y_hat)):
            prediction = self.y_hat[i]
            gold = self.y[i]

            self.confusion_matrix[prediction, gold] += 1
            for c in range(classes_count):
                self.separate_confusion_matrix[c, 1 - (prediction == c), 1 - (gold == c)] += 1

    def compute_macro_metrics(self):
        classes_count = self.confusion_matrix.shape[0]
        self.metrics_per_class = {c: {} for c in range(classes_count)}

        # precision
        total_precision = 0
        for i in range(classes_count):
            self.metrics_per_class[i]['precision'] = self.confusion_matrix[i, i] / self.confusion_matrix[i].sum()
            total_precision += self.metrics_per_class[i]['precision']
        self.macro_metrics['precision'] = total_precision / classes_count

        # recall
        total_recall = 0
        for j in range(classes_count):
            self.metrics_per_class[j]['recall'] = self.confusion_matrix[j, j] / self.confusion_matrix[:, j].sum()
            total_recall += self.metrics_per_class[j]['recall']
        self.macro_metrics['recall'] = total_recall / classes_count

        # f1-score
        total_f1 = 0
        for key in self.metrics_per_class.keys():
            self.metrics_per_class[key]['f1'] = (
                    (2 * self.metrics_per_class[key]['precision'] * self.metrics_per_class[key]['recall'])
                    / (self.metrics_per_class[key]['precision'] + self.metrics_per_class[key]['recall'])
            )
            total_f1 += self.metrics_per_class[key]['f1']
        self.macro_metrics['f1'] = total_f1 / classes_count

        table_data = []
        for c in range(classes_count):
            row = [
                f'Class {c}',
                f"{self.metrics_per_class[c]['precision']:.4f}",
                f"{self.metrics_per_class[c]['recall']:.4f}",
                f"{self.metrics_per_class[c]['f1']:.4f}"
            ]
            table_data.append(row)

        table_data.append([
            'Macro Average',
            f"{self.macro_metrics['precision']:.4f}",
            f"{self.macro_metrics['recall']:.4f}",
            f"{self.macro_metrics['f1']:.4f}"
        ])

        print(tabulate(table_data, headers=['Class', 'Precision', 'Recall', 'F1-Score'], tablefmt='grid'))


    def compute_micro_metrics(self):
        pooled_confusion_matrix = np.sum(np.copy(self.separate_confusion_matrix), axis=0)
        self.micro_metrics['precision'] = pooled_confusion_matrix[0, 0] / np.sum(pooled_confusion_matrix[0], axis=0)
        self.micro_metrics['recall'] = pooled_confusion_matrix[0, 0] / np.sum(pooled_confusion_matrix[:, 0], axis=0)
        self.micro_metrics['f1'] = (
                (2 * self.micro_metrics['precision'] * self.micro_metrics['recall'])
                / (self.micro_metrics['precision'] + self.micro_metrics['recall'])
        )

        table_data = [
            ["Precision", f"{self.micro_metrics['precision']:.4f}"],
            ["Recall", f"{self.micro_metrics['recall']:.4f}"],
            ["F1 Score", f"{self.micro_metrics['f1']:.4f}"],
        ]

        print(tabulate(table_data, headers=["Micro Metrics", "Value"], tablefmt="grid"))

    def plot_results(self):
        metrics = ['precision', 'recall', 'f1']
        x = np.arange(len(metrics))

        plt.figure(figsize=(10, 6))

        # Plot each class's metrics
        for class_id, class_metrics in self.metrics_per_class.items():
            plt.plot(
                x,
                [class_metrics[metric] for metric in metrics],
                marker='o',
                label=f'Class {class_id}'
            )

        # Plot macro average metrics with dashed line
        plt.plot(
            x,
            [self.macro_metrics[metric] for metric in metrics],
            marker='o',
            linestyle='--',
            color='black',
            label='Macro Average'
        )

        # Plot micro average metrics with dotted line
        plt.plot(
            x,
            [self.micro_metrics[metric] for metric in metrics],
            marker='o',
            linestyle=':',
            color='grey',
            label='Micro Average'
        )

        plt.xticks(x, metrics)  # Set x-ticks to the metric names
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.title("Performance Metrics by Class and Averages Schemes")
        plt.legend()
        plt.grid(True)

        plt.show()

