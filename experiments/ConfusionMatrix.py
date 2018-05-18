import math
import statistics
from collections import OrderedDict


class ConfusionMatrix:
    """
    Implementation of confusion matrix for evaluating learning algorithms; computes macro F-measure,
    accuracy, confidence intervals
    """

    def __init__(self):
        self.total = 0
        self.correct = 0
        self.map = dict()
        self.all_labels = set()

    def increase_value(self, gold_value: str, observed_value: str, times: int = 1) -> None:
        """
        Increases value of goldValue x observedValue n times
        :param gold_value: expected gold value
        :param observed_value:  actual predicted value
        :param times: how many times
        """
        self.all_labels.add(gold_value)
        self.all_labels.add(observed_value)

        if gold_value not in self.map:
            self.map[gold_value] = dict()

        if observed_value not in self.map[gold_value]:
            self.map[gold_value][observed_value] = 0

        current_value = self.map[gold_value][observed_value]
        self.map[gold_value][observed_value] = current_value + times

        self.total += times

        if gold_value == observed_value:
            self.correct += times

    def get_accuracy(self) -> float:
        return float(self.correct) / float(self.total)

    def get_total_sum(self) -> int:
        return self.total

    def get_row_sum(self, label: str) -> int:
        result = 0

        current_row = self.map[label]
        assert isinstance(current_row, dict)
        for i in current_row.values():
            result += i

        return result

    def get_col_sum(self, label: str) -> int:
        result = 0

        rows = list(self.map.items())
        for row_tuple in rows:
            row = row_tuple[1]
            assert isinstance(row, dict)
            if label in row:
                result += row[label]

        return result

    def get_precision_for_labels(self) -> dict:
        precisions = dict()

        for label in self.all_labels:
            precision = self.get_precision_for_label(label)
            precisions[label] = precision

        return OrderedDict(sorted(precisions.items(), key=lambda t: t[0]))

    def get_precision_for_label(self, label: str) -> float:
        precision = 0.0
        tp = 0
        fp_and_tp = 0

        if label in self.map and label in self.map[label]:
            tp = self.map[label][label]
            fp_and_tp = self.get_col_sum(label)

        if fp_and_tp > 0:
            precision = float(tp) / float(fp_and_tp)

        return precision

    def get_f_measure_for_labels(self, beta: float = 1) -> dict:
        """
        Returns F1 score for all labels. See http://en.wikipedia.org/wiki/F1_score
        :param beta: the beta parameter higher than 1 prefers recall, lower than 1 prefers precision
        :return: dict (label, F1)
        """
        f_measure = dict()

        precision_for_labels = self.get_precision_for_labels()
        recall_for_labels = self.get_recall_for_labels()

        for label in self.all_labels:
            p = precision_for_labels.get(label)
            r = recall_for_labels.get(label)

            fm = 0.0

            if (p + r) > 0:
                fm = (1.0 + (beta * beta)) * ((p * r) / ((beta * beta * p) + r))

            f_measure[label] = fm

        return OrderedDict(sorted(f_measure.items(), key=lambda t: t[0]))

    def get_recall_for_labels(self) -> dict:
        """
        Return recall for all labels
        :return: dict (label, recall)
        """
        recalls = dict()

        for label in self.all_labels:
            recall = self.get_recall_for_label(label)

            recalls[label] = recall

        return OrderedDict(sorted(recalls.items(), key=lambda t: t[0]))

    def get_recall_for_label(self, label: str) -> float:
        """
        Return recall for a single label
        :param label: label
        :return: recall
        """
        fn_and_tp = 0
        recall = 0.0
        tp = 0

        if label in self.map and label in self.map[label]:
            tp = self.map[label][label]
            fn_and_tp = self.get_row_sum(label)

        if fn_and_tp > 0:
            recall = float(tp) / float(fn_and_tp)

        return recall

    def get_macro_f_measure(self) -> float:
        # first macro precision
        macro_p = statistics.mean(self.get_precision_for_labels().values())
        macro_r = statistics.mean(self.get_recall_for_labels().values())

        return (2 * macro_p * macro_r) / (macro_p + macro_r)

    def get_confidence_interval_95(self) -> float:
        """
        Returns the half of the confidence interval on accuracy on alpha = 95
        :return: float
        """
        return 1.96 * math.sqrt(self.get_accuracy() * (1.0 - self.get_accuracy()) / self.total)

    def print_label_p_r_f1(self) -> str:
        precision_for_labels = self.get_precision_for_labels()
        recall_for_labels = self.get_recall_for_labels()
        f_m_for_labels = self.get_f_measure_for_labels()

        result = []

        for label in sorted(list(self.all_labels)):
            result.append("%s: P=%.4f/R=%.4f/F1=%.4f" % (
                label, precision_for_labels[label], recall_for_labels[label], f_m_for_labels[label]
            ))

        return " ".join(result)

    def __str__(self) -> str:
        rows = [[]]
        rows[0].append('↓gold pr.→')
        labels = sorted(self.all_labels)
        rows[0].extend(labels)
        rows[0].append('[sum]')

        for gold_label in labels:
            row = [gold_label]
            row_sum = 0
            for predicted_label in labels:
                if gold_label in self.map and predicted_label in self.map[gold_label]:
                    value = self.map[gold_label][predicted_label]
                    row_sum += value
                    row.append(str(value))
                else:
                    row.append(str(0))
            row.append(row_sum)
            rows.append(row)

        last_row = ["[sum]"]
        for label in labels:
            last_row.append(str(self.get_col_sum(label)))
        last_row.append(str(self.total))
        rows.append(last_row)

        result = ""

        for row in rows:
            for col in row:
                result += "%10s" % col
            result += "\n"

        return result.strip()


if __name__ == "__main__":
    cm = ConfusionMatrix()
    cm.increase_value("pos", "pos", 3)
    cm.increase_value("neg", "neg", 373)
    cm.increase_value("pos", "pos", 4)
    cm.increase_value("neg", "pos", 1)
    cm.increase_value("neg", "neg", 372)
    cm.increase_value("pos", "pos", 4)
    cm.increase_value("neg", "pos", 13)
    cm.increase_value("neg", "neg", 372)
    cm.increase_value("pos", "pos", 3)
    cm.increase_value("pos", "neg", 1)
    cm.increase_value("neg", "pos", 5)
    cm.increase_value("neg", "neg", 372)
    cm.increase_value("neg", "xxx", 100)

    print("------\nCombined confusion matrix")
    print(cm)
    print("Precision (pos): ", cm.get_precision_for_label("pos"))
    print("Recall (pos): ", cm.get_recall_for_label("pos"))
    print("F-measure (pos) F_tp,fp: ", cm.get_f_measure_for_labels().get("pos"))
    print("accuracy:", cm.get_accuracy(), "+/-", cm.get_confidence_interval_95())
    print("Macro F1:", cm.get_macro_f_measure())
    print(cm.print_label_p_r_f1())
