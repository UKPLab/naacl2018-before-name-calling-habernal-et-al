# python package: lda 1.0.5
import lda
import numpy as np
import scipy.sparse

from tcframework import TokenizedDocument, TokenizedDocumentReader
from vocabulary import Vocabulary


class LDAModel:
    """
    Wrapper for LDA model using own implementation of vocabulary; supports serialization
    """

    def __init__(self, vocabulary: Vocabulary):
        self._vocabulary = vocabulary
        self._reader_instances = []
        self._model = None

    def train(self, document_reader: TokenizedDocumentReader, topics: int, iterations: int = 1500) -> None:
        """
        Trains the LDA model; all tokens are lower-cased
        :param document_reader: text reader
        :param topics: number of topics
        :param iterations: number of iterations
        """
        self._reader_instances = document_reader.instances

        # create large sparse matrix
        vectors = [self.__instance_to_sparse_vector(i) for i in self._reader_instances]
        doc_term_matrix = scipy.sparse.vstack(vectors)

        print('Doc term matrix shape:', doc_term_matrix.shape)

        self._model = lda.LDA(n_topics=topics, n_iter=iterations, random_state=1)
        self._model.fit(doc_term_matrix)

        # self.__debug_model()

    def __instance_to_sparse_vector(self, instance: TokenizedDocument) -> scipy.sparse.coo_matrix:
        assert isinstance(instance, TokenizedDocument)
        doc_vector_len = len(self._vocabulary.word_to_index_mapping)
        doc_vector = np.zeros(doc_vector_len, dtype=np.int)
        for token in instance.tokens:
            assert isinstance(token, str)
            t = token.lower()
            # index
            index = self._vocabulary.word_to_index_mapping.get(t, self._vocabulary.word_to_index_mapping[
                self._vocabulary.oov])
            # print(t, index)
            doc_vector[index] = doc_vector[index] + 1

        # print(doc_vector)
        sparse_vector = scipy.sparse.coo_matrix(doc_vector)
        # X.append(doc_vector)
        return sparse_vector

    def __debug_model(self):
        topic_word = self._model.topic_word_
        n_top_words = 8
        print(topic_word)
        for i, topic_dist in enumerate(topic_word):
            words = []
            for w in np.argsort(topic_dist)[:-(n_top_words + 1):-1]:
                words.append(self._vocabulary.index_to_word_mapping.get(w))
            # topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            print('Topic {}: {}'.format(i, ' '.join(words)))

        # doc_topic = self._model.doc_topic_
        # for i in range(10):
        #     print(doc_topic[i].argmax())

        for i in self._reader_instances[6001:6030]:
            assert isinstance(i, TokenizedDocument)
            v = self.__instance_to_sparse_vector(i)
            distribution = self._model.transform(v)
            print(distribution.argmax())
            print(i.tokens)
            print("-----")

    def infer_probability_distribution(self, instance: TokenizedDocument) -> np.ndarray:
        v = self.__instance_to_sparse_vector(instance)
        # make from (1, 50) a (50,) vector
        distribution = self._model.transform(v).flatten()
        assert isinstance(distribution, np.ndarray)

        # make sure it's a vector
        assert len(distribution.shape) == 1

        return distribution

    def serialize(self, model_file_name: str) -> None:
        import pickle
        with open(model_file_name, "wb") as f:
            pickle.dump(self._model, f)
            f.close()
        print("LDA model serialized to %s" % model_file_name)

    def deserialize(self, model_file_name: str) -> None:
        import pickle
        with open(model_file_name, "rb") as f:
            self._model = pickle.load(f)
            assert isinstance(self._model, lda.LDA)
        print("LDA model deserialized from %s" % model_file_name)


if __name__ == '__main__':
    from regression_experiments import UnlabeledRegressionTSVReader
    voc = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')
    reader = UnlabeledRegressionTSVReader('data/experiments/controversy-unlabeled.tsv')

    model = LDAModel(voc)
    model.train(reader, 50)
    model.serialize('LDA_model_50t.pkl')
    model.deserialize('LDA_model_50t.pkl')
