from tcframework import LabeledTokenizedDocument


class AbstractTokenizedDocumentClassifier:
    def __init__(self):
        self._label_to_int_map = dict()
        self._int_to_label_map = dict()
        self._max_length = 0

    @property
    def max_length(self) -> int:
        """
        Returns max length of a sequence found in the training data
        :return: int
        """
        assert self._max_length
        return self._max_length

    def train(self, labeled_document_list: list, validation: bool = True) -> None:
        self.create_label_to_int_mapping(labeled_document_list)

        assert len(labeled_document_list) > 0, "No training documents"
        assert all([isinstance(_, LabeledTokenizedDocument) for _ in labeled_document_list])

        # find the longest document
        self._max_length = max([len(doc.tokens) for doc in labeled_document_list])
        pass

    def test(self, document_list: list, **kwargs) -> list:
        raise NotImplemented()

    def label(self, document_list: list, **kwargs) -> list:
        raise NotImplemented()

    def create_label_to_int_mapping(self, labeled_instances: list) -> None:
        """
        Collects all labels and fills mapping (integer:str), (str:integer)
        :param labeled_instances: list of LabeledTokenizedDocument instances
        """
        assert len(labeled_instances) > 0, "labeled_instances list is empty"

        all_labels = set()

        for instance in labeled_instances:
            assert isinstance(instance, LabeledTokenizedDocument)
            all_labels.add(instance.label)

        for index, label in enumerate(list(all_labels)):
            self._int_to_label_map[index] = label
            self._label_to_int_map[label] = index


class RandomTokenizedDocumentClassifier(AbstractTokenizedDocumentClassifier):
    def test(self, document_list: list, **kwargs) -> list:
        assert self._int_to_label_map
        assert isinstance(self._int_to_label_map, dict)

        import random
        # get a random "label"
        result = []
        for i in range(len(document_list)):
            label_int = random.randint(min(self._int_to_label_map.keys()), max(self._int_to_label_map.keys()))
            # get the corresponding str value
            result.append(self._int_to_label_map[label_int])

        return result


class MajorityClassTokenizedDocumentClassifier(AbstractTokenizedDocumentClassifier):
    def __init__(self):
        super().__init__()
        self._most_common_label = ''

    def train(self, labeled_document_list: list, validation: bool = True) -> None:
        super().train(labeled_document_list, validation)

        # collect frequencies
        from collections import Counter
        counter = Counter()
        for instance in labeled_document_list:
            counter.update([instance.label])

        # most common is a list of tuples (str: count)
        self._most_common_label = counter.most_common()[0][0]

    def test(self, document_list: list, *kwargs) -> list:
        # return a list of most common labels
        return [self._most_common_label] * len(document_list)


class AbstractTokenizedDocumentRegression:
    def __init__(self):
        self._max_length = 0

    @property
    def max_length(self) -> int:
        """
        Returns max length of a sequence found in the training data
        :return: int
        """
        assert self._max_length
        return self._max_length

    def train(self, labeled_document_list: list, validation: bool = True) -> None:
        assert len(labeled_document_list) > 0, "No training documents"
        assert all([isinstance(_, LabeledTokenizedDocument) for _ in labeled_document_list])

        # find the longest document
        self._max_length = max([len(doc.tokens) for doc in labeled_document_list])
        pass

    def test(self, document_list: list) -> list:
        """
        Returns a list of predicted float values (aka labels)
        :param document_list: test documents
        """
        pass

    def label(self, document_list: list) -> list:
        pass


class RandomTokenizedDocumentRegression(AbstractTokenizedDocumentRegression):
    def __init__(self):
        super().__init__()
        # get range from training data
        self._max_label_value = None
        self._min_label_value = None

    def train(self, labeled_document_list: list, validation: bool = True):
        super(RandomTokenizedDocumentRegression, self).train(labeled_document_list)

        # find the range of labels
        labels_as_float = [float(doc.label) for doc in labeled_document_list]

        self._max_label_value = max(labels_as_float)
        self._min_label_value = min(labels_as_float)

        # print("Min/max", self._min_label_value, self._max_label_value)

    def test(self, labeled_document_list: list):
        import random
        # get a random "label"
        result = []
        for i in range(len(labeled_document_list)):
            result.append(random.uniform(self._min_label_value, self._max_label_value))

        return result


class MeanValueTokenizedDocumentRegression(AbstractTokenizedDocumentRegression):
    def __init__(self):
        super().__init__()
        # get range from training data
        self._mean_value = 0

    def train(self, labeled_document_list: list, validation: bool = True):
        super(MeanValueTokenizedDocumentRegression, self).train(labeled_document_list)

        # find the range of labels
        labels_as_float = [float(doc.label) for doc in labeled_document_list]

        import statistics
        self._mean_value = statistics.mean(labels_as_float)

        print("Mean:", self._mean_value)

    def test(self, labeled_document_list: list):
        # get a random "label"
        result = []
        for i in range(len(labeled_document_list)):
            result.append(self._mean_value)

        return result
