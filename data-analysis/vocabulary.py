import collections


class Vocabulary:
    """
    Implements a (optionally size-limited) vocabulary including OOV, BOS (begin of sequence), EOS (end of
    sequence), and PAD (padding) tokens. Provides a unique mapping from word to index and from index to word;
    words are internally sorted by their frequency.
    """

    def __init__(self):
        # list of tuples (word, counts) sorted by counts ascending
        self.__vocabulary = []
        # special constant tokens: out-of-vocabulary, begin-of-sequence, end-of-sequence, padding
        self.__oov = 'OOV__'
        self.__bos = 'BOS__'
        self.__eos = 'EOS__'
        self.__pad = 'PAD__'

        # mapping words->indices and revers
        self.__word_to_index_mapping = None
        self.__index_to_word_mapping = None

    @property
    def oov(self) -> str:
        """
        Out-of-vocabulary word constant
        :return: OOV
        """
        return self.__oov

    @property
    def eos(self) -> str:
        """
        End of sequence constant
        :return: EOS
        """
        return self.__eos

    @property
    def bos(self) -> str:
        """
        Begin of sequence constant
        :return: BOS
        """
        return self.__bos

    @property
    def pad(self) -> str:
        """
        Padding constant
        :return: PAD
        """
        return self.__pad

    @property
    def vocabulary(self):
        raise Exception('Direct access to the vocabulary is not allowed, use "word_to_index_mapping" instead')

    @property
    def word_to_index_mapping(self) -> dict:
        # lazy initialization
        """
        Dictionary (word:str, index:int); MUST NOT BE MODIFIED TO REMAIN CONSISTENT!
        :return: dict
        """
        if not self.__word_to_index_mapping:
            self.__create_mappings__()

        return self.__word_to_index_mapping

    @property
    def index_to_word_mapping(self) -> dict:
        # lazy initialization
        """
        Mapping from indices to words (index:int, word:str).
        MUST NOT BE MODIFIED TO REMAIN CONSISTENT!
        :return: dict
        """
        if not self.__index_to_word_mapping:
            self.__create_mappings__()

        return self.__index_to_word_mapping

    def build_from_tokenized_documents(self, documents: list, top_k: int = -1) -> None:
        """
        Given a list of documents with tokenized texts, builds the vocabulary
        :param documents: list of LabeledTokenizedDocument instances
        :param top_k: limit vocabulary to top_k most frequent words
        """
        from tcframework import LabeledTokenizedDocument
        counter = collections.Counter()

        for document in documents:
            assert isinstance(document, LabeledTokenizedDocument)
            # update counts
            counter.update(document.tokens)

        # trunk to top K words if required
        if top_k > 0:
            self.__vocabulary = counter.most_common(top_k)
        else:
            self.__vocabulary = counter.most_common()

        # make sure it's a list of tuples
        assert isinstance(self.__vocabulary, list)
        assert isinstance(self.__vocabulary[0], tuple)
        assert len(self.__vocabulary[0]) == 2

    def build_from_text_file(self, text_file: str, max_lines: int = -1, top_k: int = -1) -> None:
        """
        Builds vocabulary from a (possibly large) text file. Each line is tokenized (see "tokenize()"
        in "preprocessing.py".
        :param text_file: Input plain text file
        :param max_lines: Maximum lines read (if negative, no limit)
        :param top_k: Final vocabulary size (if negative, no limit)
        """
        line_counter = 0

        counter = collections.Counter()

        with open(text_file) as f:
            for line in f:
                counter.update(tokenize(line))
                line_counter += 1
                if line_counter % 1000 == 0:
                    print("%d lines processed" % line_counter)
                if 0 < max_lines < line_counter:
                    print("Maximum line limit (%d) reached, reading stopped" % max_lines)
                    break

        # trunk to top K words if required
        if top_k > 0:
            self.__vocabulary = counter.most_common(top_k)
        else:
            self.__vocabulary = counter.most_common()

        # make sure it's a list of tuples
        assert isinstance(self.__vocabulary, list)
        assert isinstance(self.__vocabulary[0], tuple)
        assert len(self.__vocabulary[0]) == 2

    def __create_mappings__(self):
        # create a list and add special tokens first
        most_common_words_list = [self.pad, self.oov, self.bos, self.eos]

        # convert to a list
        for word, _ in self.__vocabulary:
            most_common_words_list.append(word)

        # init empty dictionaries
        self.__word_to_index_mapping = dict()
        self.__index_to_word_mapping = dict()

        # create a dictionary and a reverse dictionary
        for index, word in enumerate(most_common_words_list):
            self.__word_to_index_mapping[word] = index
            self.__index_to_word_mapping[index] = word

    @staticmethod
    def deserialize(gzip_file: str):
        """
        De-serialization of vocabulary form pickled file
        :param gzip_file: gz pkl file
        :return: Vocabulary instance
        """
        import gzip
        import pickle

        with gzip.open(gzip_file, 'rb') as _:
            result = pickle.load(_)
            assert isinstance(result, Vocabulary)

            return result

    @staticmethod
    def tokenize(s: str) -> list:
        """
        Tokenize the given text using TreebankWordTokenizer delivered along with NLTK
        :param s: text
        :return: list of tokens
        """
        from nltk import TreebankWordTokenizer
        import unicodedata

        tokenizer = TreebankWordTokenizer()
        tokens = tokenizer.tokenize(s)
        result = []
        for word in tokens:
            # the last "decode" function is because of Python3
            # http://stackoverflow.com/questions/2592764/what-does-a-b-prefix-before-a-python-string-mean
            w = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8').strip()
            # and add only if not empty (it happened in some data that there were empty tokens...)
            if w:
                result.append(w)

        return result

if __name__ == '__main__':
    # voc = Vocabulary()
    # voc.build_from_text_file('/mnt/hdfs/c4corpus-part-en/c4enplaintext-sample-2.5B-tokens.txt', max_lines=10000000,
    #                          top_k=100000)
    # print(len(voc.index_to_word_mapping))

    import pickle

    # with open('en-top100k.vocabulary.pkl', 'wb') as f:
    #     pickle.dump(voc, f)

    voc = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')

    import embeddings

    embeddings = embeddings.GloveEmbeddings()
    embeddings.load(voc, '/home/user-ukp/data2/glove.840B.300d.txt.gz')

    # the resulting embeddings vocabulary has 96% coverage; from the 100k top words, 4k were not found in embeddings

    with open('en-top100k.embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)


