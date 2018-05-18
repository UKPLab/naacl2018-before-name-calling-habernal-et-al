from RedditThread import RedditThread
from embeddings import WordEmbeddings
from vocabulary import Vocabulary
import numpy as np


class SemanticSimilarityHelper:
    def __init__(self):
        self.vocabulary = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')
        self.embeddings = WordEmbeddings.deserialize('en-top100k.embeddings.pkl.gz')

        assert isinstance(self.vocabulary, Vocabulary)
        assert isinstance(self.embeddings, WordEmbeddings)

        # for caching computed average word vectors (it's expensive)
        # dictionary = (str, np.ndarray)
        # key = text, value = average word vector
        self._average_word_vector_cache = dict()

    def average_embeddings_vector_thread(self, thread: RedditThread, skip_last_entry: bool = False) -> np.ndarray:
        """
        Sums up all comments into a single text and computes the average embedding vector
        :param skip_last_entry: not taking the last entry into account
        :param thread: thread
        :return: nd array
        """
        c = thread.comments
        if skip_last_entry:
            c = c[:-1]

        bodies = []
        for comment in c:
            bodies.append(comment.body)

        return self.average_embeddings_vector(" ".join(bodies))

    def average_embeddings_vector(self, text: str) -> np.ndarray:
        """
        Given a text, this method tokenizes it and computes an average embedding vector
        :param text: string
        :return: 1-D numpy array
        """
        # if cached, return the cached vector
        if text in self._average_word_vector_cache:
            return self._average_word_vector_cache[text]

        tokens = Vocabulary.tokenize(text)

        if not tokens:
            print("No tokens in text '%s' but its length was %d" % (text, len(text)))
            # probably one or more OOV words
            tokens = [self.vocabulary.word_to_index_mapping.get(self.vocabulary.oov)]
        assert tokens

        # create empty array
        result = np.zeros(self.embeddings.dimension)

        for token in tokens:
            # index of the token
            i = self.vocabulary.word_to_index_mapping.get(token)
            if not i:
                i = self.vocabulary.word_to_index_mapping.get(self.vocabulary.oov)
            # print("Word", token, "index", i)
            e = self.embeddings.word_index_embeddings_vector_dict.get(i)
            # print("Embeddings", e[0:5])

            result += e
            # print("Result", result[0:5])

        assert np.isfinite(result).all(), "%s" % result

        # compute average
        result = result / len(tokens)
        # print("Result", result[0:5])
        assert np.isfinite(result).all(), "%s" % result

        # add to the cache
        self._average_word_vector_cache[text] = result

        return result

    def average_embeddings_vector_tokens(self, tokens: list) -> np.ndarray:
        """
        Given a text, this method tokenizes it and computes an average embedding vector
        :param tokens: list
        :return: 1-D numpy array
        """
        if not tokens:
            print("No tokens in text '%s' but its length was %d" % (tokens, len(tokens)))
            # probably one or more OOV words
            tokens = [self.vocabulary.word_to_index_mapping.get(self.vocabulary.oov)]
        assert tokens

        # create empty array
        result = np.zeros(self.embeddings.dimension)

        for token in tokens:
            # index of the token
            i = self.vocabulary.word_to_index_mapping.get(token)
            if not i:
                i = self.vocabulary.word_to_index_mapping.get(self.vocabulary.oov)
            # print("Word", token, "index", i)
            e = self.embeddings.word_index_embeddings_vector_dict.get(i)
            # print("Embeddings", e[0:5])

            result += e
            # print("Result", result[0:5])

        assert np.isfinite(result).all(), "%s" % result

        # compute average
        result = result / len(tokens)
        # print("Result", result[0:5])
        assert np.isfinite(result).all(), "%s" % result

        # add to the cache
        # self._average_word_vector_cache[text] = result

        return result

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        dot = vec1.dot(vec2)

        return 1 - (dot / (norm1 * norm2))

    @staticmethod
    def distance_vec(vec1: np.ndarray, vec2: np.ndarray, vec1_len: int, vec2_len: int) -> float:
        cos_sim = SemanticSimilarityHelper.cosine_similarity(vec1, vec2)

        l_dist = max([vec1_len, vec2_len]) / min([vec1_len, vec2_len])

        return l_dist * cos_sim

    def distance(self, tokens1: list, tokens2: list) -> float:
        l1 = len(tokens1)
        l2 = len(tokens2)

        v1 = self.average_embeddings_vector_tokens(tokens1)
        v2 = self.average_embeddings_vector_tokens(tokens2)
        cos_sim = SemanticSimilarityHelper.cosine_similarity(v1, v2)

        l_dist = max([l1, l2]) / min([l1, l2])

        return l_dist * cos_sim


if __name__ == '__main__':
    similarity_helper = SemanticSimilarityHelper()
    t1 = 'This is a nice text.'
    t2 = 'This is another large text that is also very nice.'
    t3 = 'I think that this is wrong.'
    t4 = 'This is another text, quite nice.'
    t5 = 'He has a nice text.'
    # print(similarity_helper.average_embeddings_vector(t1))
    # print(similarity_helper.average_embeddings_vector(t2))
    # print(similarity_helper.average_embeddings_vector(t3))
    similarity_helper.distance(t1, t2)
    similarity_helper.distance(t1, t3)
    similarity_helper.distance(t1, t4)
    similarity_helper.distance(t1, t5)
