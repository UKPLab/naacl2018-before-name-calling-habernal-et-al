import random

import preprocessing
from classifiers import AbstractTokenizedDocumentRegression
from embeddings import WordEmbeddings
from nnclassifiers import NeuralRegressionLDA
from tcframework import LabeledTokenizedDocumentReader, LabeledTokenizedDocument, Fold, RegressionSpearmanEvaluator, \
    TokenizedDocumentReader, TokenizedDocument, AbstractEvaluator
from vocabulary import Vocabulary


class DocumentRegressionExperiment:
    def __init__(self, labeled_document_reader: LabeledTokenizedDocumentReader,
                 classifier: AbstractTokenizedDocumentRegression, evaluator: AbstractEvaluator):
        self.reader = labeled_document_reader
        self.classifier = classifier
        self.evaluator = evaluator

    def run(self) -> None:
        __folds = self.reader.get_folds()

        for i, fold in enumerate(__folds, start=1):
            assert isinstance(fold, Fold)
            assert fold.train and fold.test

            print("Running fold %d/%d" % (i, len(__folds)))
            self.classifier.train(fold.train)
            predicted_labels = self.classifier.test(fold.test)
            # self.classifier.label(fold.unlabeled)

            self.evaluate_fold(fold.test, predicted_labels)

            print("Evaluating after %d folds" % i)
            self.evaluator.evaluate()

        self.evaluator.evaluate()

    def evaluate_fold(self, labeled_document_instances: list, predicted_labels: list):
        assert labeled_document_instances
        assert len(predicted_labels)
        assert len(labeled_document_instances) == len(predicted_labels), "Prediction size mismatch"

        # convert string labels to floats
        all_gold_labels = [float(doc.label) for doc in labeled_document_instances]

        # collect IDs
        ids = [doc.id for doc in labeled_document_instances]

        self.evaluator.add_single_fold_results(all_gold_labels, predicted_labels, ids)

    def label_external(self, document_reader: TokenizedDocumentReader) -> dict:
        self.classifier.train(self.reader.train, validation=False)
        instances = document_reader.instances

        predictions = self.classifier.test(instances)

        result = dict()
        for instance, prediction in zip(instances, predictions):
            assert isinstance(instance, TokenizedDocument)
            assert isinstance(prediction, float)
            # get id and put the label to the resulting dictionary
            result[instance.id] = prediction

        return result


class RegressionTSVReader(LabeledTokenizedDocumentReader):
    """
    Reader of a TSV file for regression experiments on a document level
    """

    def read_instances(self, input_path: str, instances_limit: int = -1):
        result = []

        with open(input_path) as f:
            for line in f:
                result.append(RegressionTSVReader.line_to_instance(line))

        if instances_limit > 0:
            return result[:instances_limit]
        else:
            return result

    @staticmethod
    def line_to_instance(line: str) -> LabeledTokenizedDocument:
        split = line.split("\t")
        result = LabeledTokenizedDocument()
        result.id = split[0]
        result.label = split[2]
        result.tokens = preprocessing.tokenize(split[1])

        return result


class UnlabeledRegressionTSVReader(TokenizedDocumentReader):
    def read_instances(self, input_path: str, instances_limit: int = -1) -> list:

        result = []

        with open(input_path) as f:
            for line in f:
                result.append(UnlabeledRegressionTSVReader.line_to_instance(line))

        if instances_limit > 0:
            return result[:instances_limit]
        else:
            return result

    @staticmethod
    def line_to_instance(line: str) -> TokenizedDocument:
        split = line.split("\t")
        result = TokenizedDocument()
        result.id = split[0]
        result.tokens = preprocessing.tokenize(split[1])

        return result


def train_and_predict_extrapolations(test_only_cv=False):
    """
    Method fore training the best models on controversy and stupidity and extrapolating on unlabeled data
    """
    # random.seed(12345)
    # random.seed(123456)
    random.seed(1234567)

    import tensorflow

    sess_config = tensorflow.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    from tensorflow.python.keras.backend import set_session

    set_session(tensorflow.Session(config=sess_config))

    vocabulary = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')
    embeddings = WordEmbeddings.deserialize('en-top100k.embeddings.pkl.gz')

    from LDAModel import LDAModel
    lda_model = LDAModel(vocabulary)
    lda_model.deserialize('LDA_model_50t.pkl')

    inputs_outputs = [('data/experiments/controversy-gold.tsv',
                       'data/experiments/controversy-unlabeled.tsv',
                       'data/experiments/controversy-unlabeled-predictions-CNN+LDA.json'),
                      ('data/experiments/stupidity-gold.tsv',
                       'data/experiments/stupidity-unlabeled.tsv',
                       'data/experiments/stupidity-unlabeled-predictions-CNN+LDA.json')]

    for input_tsv, unlabeled_tsv, output_json in inputs_outputs:
        reader = RegressionTSVReader(input_tsv, True, None, None)

        e = DocumentRegressionExperiment(reader, NeuralRegressionLDA(vocabulary, embeddings, lda_model),
                                         RegressionSpearmanEvaluator())

        if test_only_cv:
            e.run()
        else:
            # get extrapolated predictions on all unlabeled data
            extrapolated_predictions = e.label_external(UnlabeledRegressionTSVReader(unlabeled_tsv))
            assert isinstance(extrapolated_predictions, dict)
            # and save them to a JSON file
            import json
            # predictions_json = "data/experiments/stupidity-unlabeled-predictions.json"
            with open(output_json, "w") as f:
                json.dump(extrapolated_predictions, f)
                print("Predictions saved to " + output_json)
                f.close()


if __name__ == '__main__':
    # you first need to train the LDA topic model: run LDAModel.py
    train_and_predict_extrapolations(True)
    train_and_predict_extrapolations()
