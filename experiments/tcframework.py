import random

from ConfusionMatrix import ConfusionMatrix


class TokenizedDocument:
    def __init__(self):
        self._id = ''
        self._tokens = []

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value: str):
        assert isinstance(value, str) and bool(value.strip()), "Wrong parameter: '%s'" % value
        self._id = value

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, value: list):
        assert [isinstance(t, str) and len(t) for t in value], "Expected list of non-empty strings"
        self._tokens = value

    def __str__(self):
        return "id='%s', tokens='%s'" % (self._id, self._tokens)


class LabeledTokenizedDocument(TokenizedDocument):
    def __init__(self):
        super().__init__()
        self._label = ''

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str):
        assert isinstance(value, str) and bool(value.strip()), "Wrong parameter type"
        self._label = value.strip()

    def __str__(self):
        return "id='%s', label='%s', tokens='%s'" % (self._id, self._label, self._tokens)


class Fold:
    def __init__(self):
        self.train = []
        self.dev = []
        self.test = []


class TokenizedDocumentReader:
    def __init__(self, input_path: str, training_instances_limit: int = -1):
        self.instances = self.read_instances(input_path, training_instances_limit)
        TokenizedDocumentReader.validate_instances(self.instances)

    @staticmethod
    def validate_instances(instances: list) -> None:
        """
        Asserts that all instances in the list have tokens, id, and are of a proper class
        :param instances: non-empty list of instances
        """
        assert len(instances) > 0, "List of instances is empty"
        for instance in instances:
            assert isinstance(instance, TokenizedDocument)

            assert isinstance(instance.tokens, list), "tokens expected to be a list but was %s" % type(instance.tokens)

            assert len(instance.tokens) > 0, "Instance contains no tokens"
            assert len(instance.id) > 0, "Instance has empty ID"

            if len(instance.tokens) == 1:
                print("Warning: instance contains only a single token: " % instance.tokens)

    def read_instances(self, input_path: str, instances_limit: int = -1) -> list:
        pass


class LabeledTokenizedDocumentReader:
    def __init__(self, input_path_train: str, cross_validation: bool = True, input_path_dev: str = None,
                 input_path_test: str = None, training_instances_limit: int = -1):
        if not cross_validation and (not input_path_test):
            raise Exception()

        self._input_path_train = input_path_train

        self.cross_validation = cross_validation

        if cross_validation:
            self.train = self.read_instances(input_path_train, training_instances_limit)
            self.validate_instances(self.train)
        else:
            self.train = self.read_instances(input_path_train, training_instances_limit)
            self.validate_instances(self.train)
            self.dev = []
            self.test = []

            if input_path_dev:
                self.dev = self.read_instances(input_path_dev)
                self.validate_instances(self.dev)

            if input_path_test:
                self.test = self.read_instances(input_path_test)
                self.validate_instances(self.test)

    @property
    def input_path_train(self):
        return self._input_path_train

    @staticmethod
    def validate_instances(instances: list) -> None:
        """
        Asserts that all instances in the list have tokens, id, and label and
        are of a proper class
        :param instances: non-empty list of instances
        """
        assert len(instances) > 0, "List of instances is empty"
        for instance in instances:
            assert isinstance(instance, LabeledTokenizedDocument)

            assert isinstance(instance.tokens, list), "tokens expected to be a list but was %s" % type(instance.tokens)

            assert len(instance.tokens) > 0, "Instance contains no tokens"
            assert len(instance.id) > 0, "Instance has empty ID"
            assert len(instance.label) > 0, "Instance has empty label"

            if len(instance.tokens) == 1:
                print("Warning: instance contains only a single token: " % instance.tokens)

    def get_folds(self) -> list:
        import math
        if self.cross_validation:
            # shuffle pseudo-randomly
            random.shuffle(self.train)

            folds_number = 10
            chunk_size = int(math.ceil(len(self.train) / folds_number))

            fold_lists = list(self.chunks(self.train, chunk_size))
            # sanity checking
            assert len(fold_lists) == folds_number
            assert all([len(_) <= chunk_size for _ in fold_lists])

            result = []

            for index_test in range(0, len(fold_lists)):
                single_fold = Fold()
                single_fold.test = fold_lists[index_test]
                single_fold.train = []

                for index_train in range(0, len(fold_lists)):
                    if index_test != index_train:
                        single_fold.train.extend(fold_lists[index_train])

                result.append(single_fold)

            return result
        else:
            single_fold = Fold()
            single_fold.train = self.train
            single_fold.dev = self.dev
            single_fold.test = self.test
            return [single_fold]

    @staticmethod
    def validate_folds(fold_list: list) -> None:
        assert len(fold_list) == 10

        complete_fold_sizes = set()

        for f in fold_list:
            assert isinstance(f, Fold)
            complete_fold_sizes.add(len(f.train) + len(f.test))
            assert len(f.train) > len(f.test)

        # all fold have equal size
        assert len(complete_fold_sizes) == 1

    @staticmethod
    def chunks(l, n):
        """
        Yield successive n-sized chunks from l
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def read_instances(self, input_path: str, instances_limit: int = -1) -> list:
        pass


class AbstractEvaluator:
    def __init__(self):
        self._all_gold_values = []
        self._all_predicted_values = []
        self._all_ids = []

    def add_single_fold_results(self, gold_values: list, predicted_values: list, ids: list) -> None:
        raise NotImplemented()

    def evaluate(self):
        raise NotImplemented()


class ClassificationEvaluator(AbstractEvaluator):
    def add_single_fold_results(self, gold_values: list, predicted_values: list, ids: list) -> None:
        assert len(gold_values) == len(predicted_values) == len(ids)

        # labels (classes) are int values
        assert any([isinstance(_, str) for _ in gold_values])
        assert any([isinstance(_, str) for _ in predicted_values])

        self._all_gold_values.extend(gold_values)
        self._all_predicted_values.extend(predicted_values)
        self._all_ids.extend(ids)

    def evaluate(self):
        cm = ConfusionMatrix()
        for gold, pred in zip(self._all_gold_values, self._all_predicted_values):
            cm.increase_value(gold, pred)
        print(cm)
        print("Accuracy: %.4f (95ppCI: %.4f)  Macro F1: %.4f" % (
            cm.get_accuracy(), cm.get_confidence_interval_95(), cm.get_macro_f_measure()))
        print(cm.print_label_p_r_f1())


class RegressionSpearmanEvaluator(AbstractEvaluator):
    def __init__(self):
        super().__init__()

    def add_single_fold_results(self, gold_values: list, predicted_values: list, ids: list) -> None:
        assert len(gold_values) == len(predicted_values) == len(ids)

        assert any([isinstance(_, float) for _ in gold_values])
        assert any([isinstance(_, float) for _ in predicted_values])

        self._all_gold_values.extend(gold_values)
        self._all_predicted_values.extend(predicted_values)
        self._all_ids.extend(ids)

    def evaluate(self) -> tuple:
        import scipy.stats

        spearman = scipy.stats.spearmanr(self._all_gold_values, self._all_predicted_values)

        print(spearman)

        return spearman
