import numpy as np

from vocabulary import Vocabulary


class WordEmbeddings:
    def __init__(self):
        self.__dimension = None
        self.__word_index_embeddings_vector_dict = dict()

    @property
    def dimension(self) -> int:
        """
        Returns the embeddings dimension
        :return: int
        """
        if not self.__dimension:
            raise Exception('dimension unknown, embeddings probably not yet initialized')
        return self.__dimension

    @property
    def word_index_embeddings_vector_dict(self) -> dict:
        """
        Returns mapping from word index (int) to embeddings (numpy ndarray)
        :return: dict(integer, np.ndarray)
        """
        return self.__word_index_embeddings_vector_dict

    def load(self, vocabulary: Vocabulary, embeddings_file_name: str) -> None:
        self.__word_index_embeddings_vector_dict = dict()

        # param checking
        if embeddings_file_name.endswith('gz'):
            import gzip
            f = gzip.open(embeddings_file_name, 'rb')
        else:
            f = open(embeddings_file_name, 'r')

            self.__dimension = None

        for line in f:
            fixed_line = line
            if isinstance(line, bytes):
                fixed_line = line.decode('utf-8').strip()

            head, vec = self.__extract_word_and_vector_from_file_line__(fixed_line)

            assert isinstance(head, str)
            assert isinstance(vec, np.ndarray)

            # what is the dimension of embeddings?
            if self.__dimension is None:
                self.__dimension = len(vec)

            if head in vocabulary.word_to_index_mapping:
                # save under the word index
                word_index = vocabulary.word_to_index_mapping[head]
                self.__word_index_embeddings_vector_dict[word_index] = vec

        # now what to do with the remaining words that have no embeddings?
        for word_index in vocabulary.index_to_word_mapping:
            if word_index not in self.__word_index_embeddings_vector_dict:
                print("Embeddings for '%s' N/A, generating random" % vocabulary.index_to_word_mapping[word_index])
                # generate random vector
                vector_rand = 2 * 0.1 * np.random.rand(self.__dimension) - 0.1
                self.__word_index_embeddings_vector_dict[word_index] = vector_rand

        # we also need to initialize embeddings for OOV, BOS, EOS, PAD
        assert bool(self.__dimension)

        # for padding we will use a zero-vector
        vector_pad = np.array([0.0] * self.__dimension)

        # for start of sequence and OOV we add random vectors
        vector_bos = 2 * 0.1 * np.random.rand(self.__dimension) - 0.1
        vector_eos = 2 * 0.1 * np.random.rand(self.__dimension) - 0.1
        vector_oov = 2 * 0.1 * np.random.rand(self.__dimension) - 0.1

        # and add them to the mapping
        self.__word_index_embeddings_vector_dict[vocabulary.word_to_index_mapping[vocabulary.pad]] = vector_pad
        self.__word_index_embeddings_vector_dict[vocabulary.word_to_index_mapping[vocabulary.bos]] = vector_bos
        self.__word_index_embeddings_vector_dict[vocabulary.word_to_index_mapping[vocabulary.eos]] = vector_eos
        self.__word_index_embeddings_vector_dict[vocabulary.word_to_index_mapping[vocabulary.oov]] = vector_oov

    def __extract_word_and_vector_from_file_line__(self, line: str) -> tuple:
        """
        Extracts a word and its embedding vector from the given text-file line
        :param line: string
        """
        raise NotImplementedError('Must be implemented in inherited classes')

    @staticmethod
    def deserialize(gzip_file: str):
        """
        De-serialization of embeddings form pickled file
        :param gzip_file: gz pkl file
        :return: WordEmbeddings instance
        """
        import gzip
        import pickle

        with gzip.open(gzip_file, 'rb') as _:
            result = pickle.load(_)
            assert isinstance(result, WordEmbeddings)

            return result

    def to_numpy_matrix(self) -> np.ndarray:
        """
        Converting embeddings to numpy 2d array: shape = (vocabulary_size, dimension)
        :return: numpy.ndarray; shape = (vocabulary_size, dimension)
        """
        buffer = []

        # for all indices 0 to vocabulary size
        for index in sorted(self.__word_index_embeddings_vector_dict):
            buffer.append(np.array(self.__word_index_embeddings_vector_dict[index], dtype=np.float32))

        result = np.asarray(buffer)

        assert result.shape[0] == len(self.word_index_embeddings_vector_dict)
        assert result.shape[1] == self.dimension

        return result


class GloveEmbeddings(WordEmbeddings):
    def __extract_word_and_vector_from_file_line__(self, line: str):
        partition = line.partition(' ')
        return partition[0], np.fromstring(partition[2], sep=' ')


class Word2VecEmbeddings(WordEmbeddings):
    @staticmethod
    def convert_word2vec_bin_to_txt(input_path: str, output_path: str) -> None:
        """
        Converts the original C-binary format of word2vec by Mikolov et al. to a txt format.
        Use this method first before extracting embeddings for your vocabulary.
        :param input_path: the 'GoogleNews-vectors-negative300.bin' file
        :param output_path: output txt file
        """
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format(input_path, binary=True)
        model.save_word2vec_format(output_path)

    def __extract_word_and_vector_from_file_line__(self, line: str):
        # "decode" function is because of Python3
        # http://stackoverflow.com/questions/2592764/what-does-a-b-prefix-before-a-python-string-mean
        split = line.decode('utf-8').split()

        # ignore the first line
        if len(split) == 2:
            return None, None

        # word
        head = split[0]
        # the rest is the embeddings vector, convert to numpy float array
        vector = np.array(split[1:]).astype(np.float)

        return head, vector
