import unicodedata


def tokenize(s: str) -> list:
    """
    Tokenize the given text using TreebankWordTokenizer delivered along with NLTK
    :param s: text
    :return: list of tokens
    """
    from nltk import TreebankWordTokenizer

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
