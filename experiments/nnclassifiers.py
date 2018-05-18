import random

from datetime import datetime
import numpy
import sys
from tensorflow.python.keras import Input
from tensorflow.python.keras._impl.keras.layers import Bidirectional
from tensorflow.python.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding, LSTM, Concatenate, Dropout
from tensorflow.python.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import sequence

from classifiers import AbstractTokenizedDocumentRegression, AbstractTokenizedDocumentClassifier
from embeddings import WordEmbeddings
from tcframework import LabeledTokenizedDocument, TokenizedDocument
from vocabulary import Vocabulary


class SimpleLSTMTokenizedDocumentClassifier(AbstractTokenizedDocumentClassifier):
    def __init__(self, vocabulary: Vocabulary, embeddings: WordEmbeddings):
        super().__init__()
        self._model = None
        self._vocabulary = vocabulary
        self._embeddings = embeddings

    @staticmethod
    def split_to_batches_by_document_lengths(labeled_document_list: list, batch_size: int = 32) -> list:
        if len(labeled_document_list) < batch_size:
            print("Requested batch size is %d but got only %d documents" % (batch_size, len(labeled_document_list)),
                  file=sys.stderr)

        # first sort all documents by their length
        s = sorted(labeled_document_list, key=lambda labeled_document: len(labeled_document.tokens))
        # one-liner to split into lists of maximum size of batch_size
        indices = [_ for _ in range(len(s))]
        result = [s[i:i + batch_size] for i in indices[::batch_size]]

        # sanity check: all except the last batch must be batch_size long
        for i in range(len(result) - 1):
            assert len(result[i]) == batch_size

        return result

    def train(self, labeled_document_list: list, validation: bool = True) -> None:
        super().train(labeled_document_list, validation)
        assert self._label_to_int_map
        assert self._max_length

        print("Max length in training data: ", self._max_length)

        # setting up parameters
        nb_epoch = 5

        # Some models need to have fully specified input thus don't support batches with variable lengths
        train_dynamic_batches = False

        if train_dynamic_batches:
            # get the compiled model
            model = self.get_model(self._embeddings.to_numpy_matrix())

            # Usually, we would pad all documents with zeros up to the length of the largest
            # document. It works fine (with masking in Keras) but training is extremely slow, as all
            # computations are done on zero inputs and thrown away. To speed things up, we manually split
            # documents into batches sorted by their length so the number of padded positions
            # in each batch is minimal.
            batches = SimpleLSTMTokenizedDocumentClassifier.split_to_batches_by_document_lengths(labeled_document_list)
            # Let's also shuffle them
            random.shuffle(batches)
            assert isinstance(batches, list)
            assert isinstance(batches[0], list)

            # now we have to run the model.fit() function "manually"
            print("Train on %d samples" % len(labeled_document_list))
            for epoch_no in range(nb_epoch):
                # measuring time
                start = datetime.now()
                print("Epoch %d/%d " % (epoch_no + 1, nb_epoch), end='', flush=True)
                batch_loss = 0
                for batch in batches:
                    # shuffle also the batch
                    random.shuffle(batch)
                    # print dots
                    print('.', end='', flush=True)

                    batch_max_length = max([len(doc.tokens) for doc in batch])

                    x_train, y_train = ConversionHelpers. \
                        convert_all_instances_to_x_y_vectors_categorical(batch, batch_max_length, self._vocabulary,
                                                                         self._label_to_int_map)
                    # print("Training model")
                    batch_loss = model.train_on_batch({"input_sequence": x_train}, y_train)

                delta = datetime.now() - start

                # print(d)
                print("- %ds - loss %.4f" % (delta.seconds, batch_loss), flush=True)

            # assign field
            self._model = model
        else:
            # get the compiled model
            # some models might need to specify the maximum length of the model
            model = self.get_model(self._embeddings.to_numpy_matrix(), max_input_length=self._max_length)

            x_train, y_train = ConversionHelpers. \
                convert_all_instances_to_x_y_vectors_categorical(labeled_document_list, self._max_length, self._vocabulary,
                                                                 self._label_to_int_map)
            # print("Training model")
            batch_loss = model.fit({"input_sequence": x_train}, y_train, epochs=nb_epoch)

            # assign field
            self._model = model

    def test(self, document_list: list, **kwargs) -> list:
        assert self._int_to_label_map
        assert isinstance(self._int_to_label_map, dict)

        # here we need only the word indices, not the labels
        x_test = ConversionHelpers.convert_all_instances_to_x_vectors(document_list, self._max_length, self._vocabulary)

        assert isinstance(self._model, Model)
        model_predict = self._model.predict({"input_sequence": x_test}, verbose=1)

        assert isinstance(model_predict, numpy.ndarray)
        # it's a 2-D array with probability predictions, e.g.:
        # [[0.37028226  0.62971777]
        # [0.4541629   0.54583716]
        # ...]
        assert len(document_list) == model_predict.shape[0]
        assert len(self._int_to_label_map) == model_predict.shape[1]

        result = ConversionHelpers.convert_predicted_prob_dist_to_label(model_predict, self._int_to_label_map)

        return result

    def get_model(self, numpy_matrix_embeddings: numpy.ndarray, **kwargs) -> Model:
        # get default parameters
        dropout = kwargs.get('dropout', 0.0)
        assert isinstance(dropout, float)
        lstm_layer_size = kwargs.get('lstm_layer_size', 64)
        assert isinstance(lstm_layer_size, int)

        # this is the placeholder tensor for the input sequences
        # input sequence is a numpy array of word indices
        # actually, we don't need to know the input sequence length, None works too
        input_sequence = Input(shape=(None,), dtype='int32', name='input_sequence')
        # this embedding layer will transform the sequences of integers
        # into vectors of size dimension of embeddings
        embedded = Embedding(numpy_matrix_embeddings.shape[0], numpy_matrix_embeddings.shape[1],
                             weights=[numpy_matrix_embeddings], mask_zero=True)(input_sequence)

        # bidirectional LSTM using the neat Keras wrapper
        lstm_output = Bidirectional(LSTM(lstm_layer_size))(embedded)

        # dropout
        dropout_layer = Dropout(dropout)(lstm_output)

        # one extra dense layer
        fully_connected = Dense(lstm_layer_size / 2, activation='relu')(dropout_layer)

        # classification
        output = Dense(2, activation='softmax')(fully_connected)
        model = Model(inputs=[input_sequence], outputs=output)

        print("Compiling model")
        model.compile('adam', loss=categorical_crossentropy)

        return model


class StackedLSTMTokenizedDocumentClassifier(SimpleLSTMTokenizedDocumentClassifier):
    def get_model(self, numpy_matrix_embeddings: numpy.ndarray, **kwargs) -> Model:
        # get default parameters
        dropout = kwargs.get('dropout', 0.9)
        assert isinstance(dropout, float)
        lstm_layer_size = kwargs.get('lstm_layer_size', 64)
        assert isinstance(lstm_layer_size, int)

        # this is the placeholder tensor for the input sequences
        # input sequence is a numpy array of word indices
        input_sequence = Input(shape=(None,), dtype='int32', name='input_sequence')
        # this embedding layer will transform the sequences of integers
        # into vectors of size dimension of embeddings
        embedded = Embedding(numpy_matrix_embeddings.shape[0], numpy_matrix_embeddings.shape[1],
                             weights=[numpy_matrix_embeddings], mask_zero=True)(input_sequence)

        # bidirectional LSTM using the neat Keras wrapper
        lstm_output1 = Bidirectional(LSTM(lstm_layer_size, return_sequences=True))(embedded)

        # stack the next one
        lstm_output2 = Bidirectional(LSTM(lstm_layer_size))(lstm_output1)

        # dropout
        dropout_layer = Dropout(dropout)(lstm_output2)

        # one extra dense layer
        fully_connected = Dense(lstm_layer_size / 2, activation='relu')(dropout_layer)

        # classification
        output = Dense(2, activation='softmax')(fully_connected)
        model = Model(inputs=[input_sequence], outputs=output)

        print("Compiling model")
        model.compile('adam', loss=categorical_crossentropy)

        return model


class CNNTokenizedDocumentClassifier(SimpleLSTMTokenizedDocumentClassifier):
    def get_model(self, numpy_matrix_embeddings: numpy.ndarray, **kwargs) -> Model:
        # get default parameters
        dropout = kwargs.get('dropout', 0.9)
        assert isinstance(dropout, float)

        # this is the placeholder tensor for the input sequences
        # input sequence is a numpy array of word indices
        input_sequence = Input(shape=(None,), dtype='int32', name='input_sequence')
        # this embedding layer will transform the sequences of integers
        # into vectors of size dimension of embeddings
        # Conv1D does not support masking
        embedded = Embedding(numpy_matrix_embeddings.shape[0], numpy_matrix_embeddings.shape[1],
                             weights=[numpy_matrix_embeddings], mask_zero=False)(input_sequence)

        # 4 convolution layers (500 filters each)
        nb_filters = 500
        kernel_sizes = [5, 7, 9, 11]
        cnn = [Conv1D(filters=nb_filters, kernel_size=kernel_size, padding="same")(embedded) for kernel_size in
               kernel_sizes]

        # concatenate
        concatenated_cnn_outputs = Concatenate()([c for c in cnn])
        # max pooling
        pooled = GlobalMaxPooling1D()(concatenated_cnn_outputs)

        # after_dropout = Dropout(dropout)(pooled)
        # just for fun - no regularization... and it worked!
        after_dropout = Dropout(dropout)(pooled)

        # batch normalization? No, worse performance... 0.43321227654613681
        # batch_normalization = BatchNormalization()(after_dropout)

        # classification
        output = Dense(2, activation='softmax')(after_dropout)
        model = Model(inputs=[input_sequence], outputs=output)

        print("Compiling model")
        model.compile('adam', loss=categorical_crossentropy)

        return model


class ConversionHelpers:
    @staticmethod
    def convert_single_instance_to_x_vector(doc: TokenizedDocument, vocabulary: Vocabulary) -> numpy.ndarray:
        """
        Converts a single instance to a X vector (word indices)
        :param vocabulary: vocabulary
        :param doc: document instance
        :return: x vector
        """
        assert isinstance(doc, TokenizedDocument)

        # collect a list of word indices
        x_word_indices = numpy.array(
            [vocabulary.word_to_index_mapping.get(w, vocabulary.word_to_index_mapping[vocabulary.oov])
             for w in doc.tokens], dtype=numpy.int32)

        return x_word_indices

    @staticmethod
    def convert_single_instance_to_y_vector_regression(doc: LabeledTokenizedDocument):
        y_prediction = float(doc.label)

        return y_prediction

    @staticmethod
    def convert_single_instance_to_y_vector_categorical(doc: LabeledTokenizedDocument,
                                                        label_to_int_map: dict) -> numpy.ndarray:
        """
        Converts a single instance to y vector (one-hot encoding vector of classes)
        :param label_to_int_map: mapping from labels (str) to integers (0 to number of classes -1)
        :param doc: document instance
        :return: one-hot vector
        """
        assert isinstance(label_to_int_map, dict)
        # there's more than one class
        assert len(label_to_int_map) > 1

        # empty vector first
        y_prediction = numpy.zeros(len(label_to_int_map))
        # then set the label's position to 1
        y_prediction[label_to_int_map[doc.label]] = 1

        # print(y_prediction)

        return y_prediction

    @staticmethod
    def convert_single_instance_to_x_y_vectors_regression(doc: LabeledTokenizedDocument,
                                                          vocabulary: Vocabulary) -> tuple:
        """
        Converts a single instance to a X vector (word indices) and y float value (the 'label' value)
        :param vocabulary: vocabulary
        :param doc: document instance
        :return: a tuple (x vector, y value)
        """
        assert isinstance(doc, LabeledTokenizedDocument)

        # collect a list of word indices
        x_word_indices = ConversionHelpers.convert_single_instance_to_x_vector(doc, vocabulary)
        y_prediction = ConversionHelpers.convert_single_instance_to_y_vector_regression(doc)

        return x_word_indices, y_prediction

    @staticmethod
    def convert_single_instance_to_x_y_vectors_categorical(doc: LabeledTokenizedDocument,
                                                           vocabulary: Vocabulary, label_to_int_map: dict) -> tuple:
        """
        Converts a single instance to a X vector (word indices) and y vector (one-hot encoding vector of classes)
        :param label_to_int_map: mapping from labels (str) to integers (0 to number of classes -1)
        :param vocabulary: vocabulary
        :param doc: document instance
        :return: a tuple (x vector, y one-hot vector)
        """
        # collect a list of word indices
        x_word_indices = ConversionHelpers.convert_single_instance_to_x_vector(doc, vocabulary)
        # and the one-hot vector
        y_prediction = ConversionHelpers.convert_single_instance_to_y_vector_categorical(doc, label_to_int_map)

        return x_word_indices, y_prediction

    @staticmethod
    def convert_all_instances_to_x_y_vectors_categorical(instances: list, max_length: int,
                                                         vocabulary: Vocabulary,
                                                         label_to_int_map: dict) -> tuple:
        """
        Convert a list of LabeledTokenizedDocument instances to a 2-d matrix
        with padded word indices for each instance and a matrix of one-hot encoding labels for each instance
        :param label_to_int_map: mapping from labels (str) to integers (0 to number of classes -1)
        :param max_length: max length for trimming/padding
        :param vocabulary: vocabulary
        :param instances: list of LabeledTokenizedDocument instances
        :return: tuple: X matrix with shape = (number of instances, maximal document length);
        y vector shape = (number of instances, number of classes)
        """
        result_y = []

        for instance in instances:
            y = ConversionHelpers.convert_single_instance_to_y_vector_categorical(instance, label_to_int_map)
            result_y.append(y)

        # y values are int (one-hot vectors)
        result_y_numpy_vector = numpy.array(result_y, dtype=numpy.int)
        # print("Y vectors shape", result_y_numpy_vector.shape)

        # pad training instances
        result_x_numpy_array_padded = ConversionHelpers.convert_all_instances_to_x_vectors(instances, max_length,
                                                                                           vocabulary)
        return result_x_numpy_array_padded, result_y_numpy_vector

    @staticmethod
    def convert_all_instances_to_x_vectors(instances: list, max_length: int, vocabulary: Vocabulary) -> numpy.ndarray:
        """
        Convert a list of LabeledTokenizedDocument instances to a 2-d matrix
        with padded word indices for each instance and a vector of values for each instance
        :param vocabulary: vocabulary
        :param max_length: maximum sequence length (for proper padding)
        :param instances: list of LabeledTokenizedDocument instances
        :return: tuple: X matrix with shape = (number of instances, maximal document length);
        y vector shape = (number of instances)
        """
        result_x = []

        for instance in instances:
            word_indices = ConversionHelpers.convert_single_instance_to_x_vector(instance, vocabulary)
            result_x.append(word_indices)

        result_x_numpy_array = numpy.array(result_x)

        # pad training instances
        result_x_numpy_array_padded = sequence.pad_sequences(result_x_numpy_array, maxlen=max_length,
                                                             padding='post', truncating='post')

        # print("Instances shape", result_x_numpy_array_padded.shape)

        return result_x_numpy_array_padded

    @staticmethod
    def convert_all_instances_to_x_y_vectors_regression(instances: list, max_length: int,
                                                        vocabulary: Vocabulary) -> tuple:
        """
        Convert a list of LabeledTokenizedDocument instances to a 2-d matrix
        with padded word indices for each instance and a vector of values for each instance
        :param max_length: max length for trimming/padding
        :param vocabulary: vocabulary
        :param instances: list of LabeledTokenizedDocument instances
        :return: tuple: X matrix with shape = (number of instances, maximal document length);
        y vector shape = (number of instances)
        """
        result_y = []

        for instance in instances:
            y = ConversionHelpers.convert_single_instance_to_y_vector_regression(instance)
            result_y.append(y)

        # y values are float (for regression)
        result_y_numpy_vector = numpy.array(result_y, dtype=numpy.float32)

        # pad training instances
        result_x_numpy_array_padded = ConversionHelpers.convert_all_instances_to_x_vectors(instances, max_length,
                                                                                           vocabulary)

        print("Instances shape", result_x_numpy_array_padded.shape)

        return result_x_numpy_array_padded, result_y_numpy_vector

    @staticmethod
    def convert_predicted_prob_dist_to_label(y_predictions: numpy.ndarray, int_to_label_map: dict) -> list:
        # must be a two-dimensional array
        assert y_predictions.ndim == 2
        assert isinstance(y_predictions, numpy.ndarray)

        # it's a 2-D array with probability predictions, e.g.:
        # [[0.37028226  0.62971777]
        # [0.4541629   0.54583716]
        # ...]
        # make sure the cols correspond to number of classes
        assert len(int_to_label_map) == y_predictions.shape[1]

        # convert to a list of column indices with maximum probability in each row
        arg_max_indices = numpy.argmax(y_predictions, axis=-1)
        assert isinstance(arg_max_indices, numpy.ndarray)

        # and convert to a list of labels
        result = [int_to_label_map[index] for index in arg_max_indices]

        return result


class NeuralRegression(AbstractTokenizedDocumentRegression):
    def __init__(self, vocabulary: Vocabulary, embeddings: WordEmbeddings):
        super().__init__()
        self._model = None
        self._vocabulary = vocabulary
        self._embeddings = embeddings

    def get_model(self, input_length: int, numpy_matrix_embeddings: numpy.ndarray, **kwargs) -> Model:
        # get default parameters
        dropout = kwargs.get('dropout', 0.9)
        assert isinstance(dropout, float)

        # this is the placeholder tensor for the input sequences
        # input sequence is a numpy array of word indices
        input_sequence = Input(shape=(input_length,), dtype='int32')
        # this embedding layer will transform the sequences of integers
        # into vectors of size dimension of embeddings
        # Conv1D does not support masking
        embedded = Embedding(numpy_matrix_embeddings.shape[0], numpy_matrix_embeddings.shape[1],
                             input_length=input_length, weights=[numpy_matrix_embeddings], mask_zero=False)(
            input_sequence)

        # 4 convolution layers (500 filters each)
        nb_filters = 500
        kernel_sizes = [3, 5, 7, 9]
        cnn = [Conv1D(filters=nb_filters, kernel_size=kernel_size, padding="same")(embedded) for kernel_size in
               kernel_sizes]

        # concatenate
        concatenated_cnn_outputs = Concatenate()([c for c in cnn])
        # max pooling
        pooled = GlobalMaxPooling1D()(concatenated_cnn_outputs)

        # after_dropout = Dropout(dropout)(pooled)
        # just for fun - no regularization... and it worked!
        after_dropout = Dropout(0)(pooled)

        # batch normalization? No, worse performance... 0.43321227654613681
        # batch_normalization = BatchNormalization()(after_dropout)

        output_layer = Dense(1, activation='linear')(after_dropout)
        model = Model(inputs=input_sequence, outputs=output_layer)

        print("Compiling model")
        model.compile('rmsprop', mean_squared_error)

        return model

    def train(self, labeled_document_list: list, validation: bool = True):
        super(NeuralRegression, self).train(labeled_document_list)
        assert self._max_length

        print("Max length in training data: ", self._max_length)

        # setting up parameters
        nb_epoch = 20

        # get the compiled model
        model = self.get_model(self.max_length, self._embeddings.to_numpy_matrix())

        x_train, y_train = ConversionHelpers.convert_all_instances_to_x_y_vectors_regression(labeled_document_list,
                                                                                             self.max_length,
                                                                                             self._vocabulary)

        print("Training model")
        validation_split = 0.1 if validation else 0
        model.fit([x_train], y_train, epochs=nb_epoch, validation_split=validation_split, verbose=1)

        # assign field
        self._model = model

    def test(self, labeled_document_list: list) -> list:
        # here we need only the word indices, not the labels
        x_test = ConversionHelpers.convert_all_instances_to_x_vectors(labeled_document_list, self.max_length,
                                                                      self._vocabulary)

        assert isinstance(self._model, Model)
        model_predict = self._model.predict(x_test, verbose=1)

        assert isinstance(model_predict, numpy.ndarray)

        # re-type to python float and return
        return [float(_) for _ in model_predict]


class NeuralRegressionLDA(NeuralRegression):
    from LDAModel import LDAModel

    def __init__(self, vocabulary: Vocabulary, embeddings: WordEmbeddings, lda_model: LDAModel):
        super().__init__(vocabulary, embeddings)

        from LDAModel import LDAModel
        assert isinstance(lda_model, LDAModel)
        self._lda_model = lda_model

    def get_model(self, input_length: int, numpy_matrix_embeddings: numpy.ndarray, **kwargs):
        # get the number of topics
        lda_vector_size = kwargs.get('lda_vector_size', 0)
        assert lda_vector_size, "lda_vector_size not specified"

        # this is the placeholder tensor for the input sequences
        # input sequence is a numpy array of word indices

        input_sequence = Input(shape=(input_length,), name="input_sequence", dtype='int32')
        input_lda = Input(shape=(lda_vector_size,), name="input_lda", dtype='float32')

        # input_sequence = Input(shape=(input_length,), dtype='int32')
        # input_lda = Input(shape=(lda_vector_size,), dtype='float32')
        # this embedding layer will transform the sequences of integers
        # into vectors of size dimension of embeddings
        # Conv1D does not support masking
        embedded = Embedding(numpy_matrix_embeddings.shape[0], numpy_matrix_embeddings.shape[1],
                             input_length=input_length, weights=[numpy_matrix_embeddings], mask_zero=False)(
            input_sequence)

        # 4 convolution layers (500 filters each)
        nb_filters = 500
        kernel_sizes = [3, 5, 7, 9]
        cnn = [Conv1D(filters=nb_filters, kernel_size=kernel_size, padding="same")(embedded) for kernel_size in
               kernel_sizes]

        # concatenate
        concatenated_cnn_outputs = Concatenate()([c for c in cnn])
        # max pooling
        pooled = GlobalMaxPooling1D()(concatenated_cnn_outputs)

        # after_dropout = Dropout(dropout)(pooled)
        # just for fun - no regularization... and it worked!
        after_dropout = Dropout(0)(pooled)

        # batch normalization? No, worse performance... 0.43321227654613681
        # batch_normalization = BatchNormalization()(after_dropout)

        # now concatenate the LDA vector with output from CNN
        cnn_and_lda = Concatenate()([after_dropout, input_lda])

        output_layer = Dense(1, activation='linear')(cnn_and_lda)
        model = Model(inputs=[input_sequence, input_lda], outputs=output_layer)

        print("Compiling model")
        model.compile('rmsprop', mean_squared_error)

        return model

    def train(self, labeled_document_list: list, validation: bool = True):
        super(NeuralRegression, self).train(labeled_document_list)
        assert self._max_length

        print("Max length in training data: ", self._max_length)

        # setting up parameters
        nb_epoch = 20

        # get the compiled model
        model = self.get_model(self.max_length, self._embeddings.to_numpy_matrix(), lda_vector_size=50)

        x_train, y_train = ConversionHelpers.convert_all_instances_to_x_y_vectors_regression(labeled_document_list,
                                                                                             self.max_length,
                                                                                             self._vocabulary)
        x_train_lda = self.convert_all_instances_to_x_vectors_lda(labeled_document_list)

        print("Training model")
        validation_split = 0.1 if validation else 0
        model.fit({"input_sequence": x_train, "input_lda": x_train_lda}, y_train, epochs=nb_epoch,
                  validation_split=validation_split, verbose=1)

        # assign field
        self._model = model

    def test(self, labeled_document_list: list) -> list:
        # here we need only the word indices, not the labels
        x_test = ConversionHelpers.convert_all_instances_to_x_vectors(labeled_document_list, self.max_length,
                                                                      self._vocabulary)
        # and the LDA vectors
        x_test_lda = self.convert_all_instances_to_x_vectors_lda(labeled_document_list)

        assert isinstance(self._model, Model)
        model_predict = self._model.predict({"input_sequence": x_test, "input_lda": x_test_lda}, verbose=1)

        assert isinstance(model_predict, numpy.ndarray)

        # re-type to python float and return
        return [float(_) for _ in model_predict]

    def convert_all_instances_to_x_vectors_lda(self, instances: list):
        result_x = []

        for instance in instances:
            x = self._lda_model.infer_probability_distribution(instance)
            result_x.append(x)

        result_x_numpy_array = numpy.array(result_x)

        return result_x_numpy_array


class NeuralRegressionLSTM(NeuralRegression):
    def __init__(self, vocabulary: Vocabulary, embeddings: WordEmbeddings):
        super().__init__(vocabulary, embeddings)

    def get_model(self, input_length: int, numpy_matrix_embeddings: numpy.ndarray, **kwargs):
        # get default parameters
        lstm_layer_size = kwargs.get('lstm_layer_size', 64)
        assert isinstance(lstm_layer_size, int)

        dropout_rate = kwargs.get('dropout', 0.9)
        assert isinstance(dropout_rate, float)

        # this is the placeholder tensor for the input sequences
        # input sequence is a numpy array of word indices
        input_sequence = Input(shape=(input_length,), name="input_sequence", dtype='int32')

        # this embedding layer will transform the sequences of integers
        # into vectors of size dimension of embeddings
        # mask_zero=True is very important!!
        embedded = Embedding(numpy_matrix_embeddings.shape[0], numpy_matrix_embeddings.shape[1],
                             input_length=input_length, weights=[numpy_matrix_embeddings], mask_zero=True)(
            input_sequence)

        # apply forwards and backward LSTM
        forwards = LSTM(lstm_layer_size)(embedded)
        backwards = LSTM(lstm_layer_size, go_backwards=True)(embedded)

        # concatenate the outputs of the 2 LSTMs; this is the new API; and add the LDA topics
        lstm_output = Concatenate()([forwards, backwards])

        # dropout
        dropout = Dropout(dropout_rate)(lstm_output)

        # one extra dense layer
        fully_connected = Dense(32, activation='relu')(dropout)

        # linear regression
        output = Dense(1, activation='linear')(fully_connected)
        model = Model(inputs=[input_sequence], outputs=output)

        print("Compiling model")
        model.compile('rmsprop', mean_squared_error)

        return model


class NeuralRegressionLSTMLDA(NeuralRegressionLDA):
    from LDAModel import LDAModel

    def __init__(self, vocabulary: Vocabulary, embeddings: WordEmbeddings, lda_model: LDAModel):
        super().__init__(vocabulary, embeddings, lda_model)

    def get_model(self, input_length: int, numpy_matrix_embeddings: numpy.ndarray, **kwargs):
        # get default parameters
        lstm_layer_size = kwargs.get('lstm_layer_size', 64)
        assert isinstance(lstm_layer_size, int)

        dropout_rate = kwargs.get('dropout', 0.9)
        assert isinstance(dropout_rate, float)

        # get the number of topics
        lda_vector_size = kwargs.get('lda_vector_size', 0)
        assert isinstance(lda_vector_size, int)
        assert lda_vector_size, "lda_vector_size not specified"

        # this is the placeholder tensor for the input sequences
        # input sequence is a numpy array of word indices
        input_sequence = Input(shape=(input_length,), name="input_sequence", dtype='int32')
        input_lda = Input(shape=(lda_vector_size,), name="input_lda", dtype='float32')

        # this embedding layer will transform the sequences of integers
        # into vectors of size dimension of embeddings
        # mask_zero=True is very important!!
        embedded = Embedding(numpy_matrix_embeddings.shape[0], numpy_matrix_embeddings.shape[1],
                             input_length=input_length, weights=[numpy_matrix_embeddings], mask_zero=True)(
            input_sequence)

        # apply forwards and backward LSTM
        forwards = LSTM(lstm_layer_size)(embedded)
        backwards = LSTM(lstm_layer_size, go_backwards=True)(embedded)

        # concatenate the outputs of the 2 LSTMs; this is the new API; and add the LDA topics
        lstm_output = Concatenate()([forwards, backwards])

        # dropout
        dropout = Dropout(dropout_rate)(lstm_output)

        # one extra dense layer
        fully_connected = Dense(32, activation='relu')(dropout)

        merged_with_lda = Concatenate()([fully_connected, input_lda])

        # linear regression
        output = Dense(1, activation='linear')(merged_with_lda)
        model = Model(inputs=[input_sequence, input_lda], outputs=output)

        print("Compiling model")
        model.compile('rmsprop', mean_squared_error)

        return model

# if __name__ == "__main__":
#     a = numpy.asarray([[0.31982186, 0.68017817],
#                        [0.56077957, 0.43922043],
#                        [0.70330411, 0.29669583],
#                        [0.37873262, 0.62126738],
#                        [0.45111254, 0.54888743],
#                        [0.33458397, 0.665416],
#                        [0.32460552, 0.67539448]])
#     print(a)
#     print(ConversionHelpers.convert_predicted_prob_dist_to_label(a, {0: "AH", 1: "None"}))
