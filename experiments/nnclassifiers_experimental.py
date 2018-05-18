# Implementation of Structured Self Attentive Sentence Embedding as reported in:
#
# Lin, Z., Feng, M., Nogueira dos Santos, C., Yu, M., Xiang, B., Zhou, B., & Bengio, Y. (2017). A Structured
# Self-attentive Sentence Embedding. In Proceedings of the 5th International Conference on Learning Representations
# (ICLR) (pp. 1–15). Toulon, France. Retrieved from http://arxiv.org/abs/1703.03130
#
import json
import os

import numpy
import tensorflow
from tensorflow import Tensor
from tensorflow.python.keras import Input
from tensorflow.python.keras._impl.keras import backend
from tensorflow.python.keras._impl.keras.layers import TimeDistributed, Lambda
from tensorflow.python.keras._impl.keras.models import Sequential
from tensorflow.python.keras._impl.keras.regularizers import Regularizer
from tensorflow.python.keras._impl.keras.utils.layer_utils import print_summary
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import LSTM, Dense, Bidirectional, Embedding
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.models import Model

from embeddings import WordEmbeddings
from nnclassifiers import SimpleLSTMTokenizedDocumentClassifier, ConversionHelpers
from tcframework import TokenizedDocument, LabeledTokenizedDocument
from vocabulary import Vocabulary


class PRegularizer(Regularizer):
    """
    Re-implementation of the P regularizer (equation 8) from the paper
    """

    def __init__(self, coefficient=1.):
        self.coefficient = coefficient

    def __call__(self, x):
        regularization = 0.
        # for now, it's a l1 regularizer (just to try it out)
        # raise Exception
        # so what do we have here?
        # print(type(x))
        # <class 'tensorflow.python.framework.ops.Tensor'>
        assert isinstance(x, Tensor)
        # print(x.shape)
        # (?, 347, 3)
        # So it's the A matrix (after softmax)

        # we get the 3x3 (r x r) matrix
        a_times_a_transposed = tensorflow.matmul(x, x, transpose_a=True)
        # must be a Tensor
        assert isinstance(a_times_a_transposed, Tensor)
        # s = a_times_a_transposed.shape
        # print(s)
        # assert isinstance(s, TensorShape)
        # print(s.dims)
        # print(type(s.dims))
        # last_dim = s.dims[-1]
        # assert isinstance(last_dim, Dimension)
        # print(last_dim.value)
        # print(type(last_dim))

        # get the last dimension (r)
        i_matrix_dimension = a_times_a_transposed.shape.dims[-1].value

        identity_matrix = tensorflow.eye(i_matrix_dimension, i_matrix_dimension)
        assert isinstance(identity_matrix, Tensor)
        # print(identity_matrix)

        subtracted = tensorflow.subtract(a_times_a_transposed, identity_matrix)
        # print(subtracted.shape)

        frobenius_norm_squared = tensorflow.square(tensorflow.norm(subtracted, ord='fro', axis=[-2, -1]))
        # print(frobenius_norm_squared.shape)

        return backend.sum(self.coefficient * frobenius_norm_squared)
        # regularization += backend.sum(self.coefficient * backend.abs(x))
        # return regularization

    def get_config(self):
        return {'coefficient': float(self.coefficient)}


class StructuredSelfAttentiveSentenceEmbedding(SimpleLSTMTokenizedDocumentClassifier):

    def __init__(self, vocabulary: Vocabulary, embeddings: WordEmbeddings, output_dir_data_analysis: str = None):
        super().__init__(vocabulary, embeddings)
        self._output_dir_data_analysis = output_dir_data_analysis

        if not os.path.exists(output_dir_data_analysis):
            os.makedirs(output_dir_data_analysis)

    def get_model(self, numpy_matrix_embeddings: numpy.ndarray, **kwargs) -> Model:
        # get default parameters
        dropout = kwargs.get('dropout', 0.9)
        assert isinstance(dropout, float)
        lstm_layer_size = kwargs.get('lstm_layer_size', 64)
        assert isinstance(lstm_layer_size, int)

        # we also need the full input length
        # nope, we don't need it :)
        # max_input_length = kwargs.get('max_input_length')
        # assert max_input_length

        # this is the standard placeholder tensor for the input sequences; a numpy array of word indices
        # input_sequence = Input(shape=(max_input_length,), dtype='int32', name='input_sequence')
        input_sequence = Input(shape=(None,), dtype='int32', name='input_sequence')
        print("Embeddings shape:", numpy_matrix_embeddings.shape)

        # The embedding layer will transform the sequences of integers into vectors of size dimension of embeddings
        # We also fix the embeddings (not trainable)
        #
        # For this model, we have to disable masking as the Reshape layers do not support it
        embedded = Embedding(numpy_matrix_embeddings.shape[0], numpy_matrix_embeddings.shape[1],
                             weights=[numpy_matrix_embeddings], mask_zero=True, name='embeddings',
                             trainable=False)(input_sequence)

        # bidirectional LSTM using the neat Keras wrapper
        lstm_output = Bidirectional(LSTM(lstm_layer_size, return_sequences=True), name="BiLSTM")(embedded)

        # matrix H
        # maybe we don't need to reshape? -- nope, we don't need to reshape :)
        # matrix_h = Reshape((max_input_length, lstm_layer_size * 2), name='H_matrix')(lstm_output)
        matrix_h = lstm_output

        # matrix H transposed; we don't need it, it can be transposed in matmul later on
        # matrix_h_t = Lambda(lambda x: tensorflow.transpose(x, perm=[0, 2, 1]), name='HT_matrix')(matrix_h)

        # 150 was default
        param_da = 300

        output_tanh_ws1_ht = Dense(units=param_da, activation='tanh', use_bias=False, name='tanh_Ws1_HT')(matrix_h)

        # 30 was ok
        param_r = 50

        annotation_matrix_a_linear = Dense(units=param_r, activation='linear', use_bias=False, name='A_matrix')(
            output_tanh_ws1_ht)

        # 0.1 was ok too
        # the longer the coefficient, the more only blocks at the beginning are shown
        # penalization_coefficient = 0.5
        # penalization_coefficient = 1
        # this kills the performance :(
        # so without penalization we get to almost 80 percent!! :)
        penalization_coefficient = 0

        # now do the softmax over rows, not the columns
        annotation_matrix_a = Lambda(lambda x: tensorflow.nn.softmax(x, dim=1), name='A_softmax',
                                     activity_regularizer=PRegularizer(penalization_coefficient))(
            annotation_matrix_a_linear)

        # multiplication: easier to use Lambda than Multiply
        matrix_m = Lambda(lambda x: tensorflow.matmul(x[0], x[1], transpose_a=True), name='M_matrix')(
            [annotation_matrix_a, matrix_h])

        # and flatten
        dense_representation = Flatten()(matrix_m)

        # classification
        output = Dense(2, activation='softmax', name="Output_dense")(dense_representation)
        model = Model(inputs=[input_sequence], outputs=output)

        print_summary(model)

        model.compile('adam', loss=categorical_crossentropy)
        print("Model compiled")

        debug_graph = False
        if debug_graph:
            from tensorflow.python.keras.utils import plot_model
            plot_model(model, to_file='/tmp/model.png', show_shapes=True)

        return model

    def test(self, document_list: list, **kwargs) -> list:
        result = super().test(document_list)

        # here we need only the word indices, not the labels
        x_test = ConversionHelpers.convert_all_instances_to_x_vectors(document_list, self._max_length, self._vocabulary)

        # do we want to output the learned data?
        if self._output_dir_data_analysis and kwargs.get('fold_no'):
            fold_no = int(kwargs.get('fold_no'))
            # output file for the fold
            output_file = os.path.join(self._output_dir_data_analysis, "fold%d.json" % fold_no)

            # build a small model to get outputs of the alpha_vector layer
            layer_name = 'A_softmax'
            intermediate_layer_model = Model(inputs=self._model.input, outputs=self._model.get_layer(layer_name).output)
            a_matrix_output = intermediate_layer_model.predict(x_test)

            collected_words_and_weights = []

            # print(a_matrix_output)
            # for each matrix (each instance)
            for instance_index in range(len(document_list)):
                single_a_matrix = a_matrix_output[instance_index]
                single_document = document_list[instance_index]
                assert isinstance(single_document, LabeledTokenizedDocument)

                # print(single_a_matrix.shape)
                # 347, 30

                # sum over the 30 (r dimension)
                assert isinstance(single_a_matrix, numpy.ndarray)
                not_normalized_sum = single_a_matrix.sum(axis=1)

                # print(not_normalized_sum.shape)
                # print(not_normalized_sum)

                # now extract only the first n weights (n = document length)
                not_normalized_token_weights = not_normalized_sum[0:len(single_document.tokens)]
                # print(not_normalized_token_weights)

                tokens, weights = self.debug_weights_with_words(single_document, not_normalized_token_weights)

                # create a single map for this instance
                single_instance_words_and_weights = dict()
                single_instance_words_and_weights['words'] = tokens
                single_instance_words_and_weights['weights'] = weights
                single_instance_words_and_weights['id'] = single_document.id
                single_instance_words_and_weights['gold'] = single_document.label
                single_instance_words_and_weights['predicted'] = result[instance_index]
                # add to the result
                collected_words_and_weights.append(single_instance_words_and_weights)

            # save the collected data to a json file
            with open(output_file, 'w') as f:
                json.dump(collected_words_and_weights, f)
                f.close()
                print("Collected words+weights saved to %s" % output_file)

            # check the softmax of a random column
            # matrix_for_single_instance = a_matrix_output[0]
            # print(matrix_for_single_instance)
            # assert isinstance(matrix_for_single_instance, numpy.ndarray)
            # print(matrix_for_single_instance.shape)
            # single_column_softmaxed = matrix_for_single_instance[:, 0]
            # print(single_column_softmaxed)
            # assert isinstance(single_column_softmaxed, numpy.ndarray)
            # the sum should be 1.0
            # print(single_column_softmaxed.sum())
            # ok, that worked!! :)

        # intermediate_layer_model = Model(inputs=self._model.input,
        #                                  outputs=self._model.get_layer('Reshaped_BiLSTM_2').output)
        # lstm2_output = intermediate_layer_model.predict(x_test)
        # print(lstm2_output)
        #
        # self.debug_weights_with_words(document_list, a_matrix_output)

        return result

    def debug_weights_with_words(self, doc: TokenizedDocument, weights: numpy.ndarray) -> tuple:
        # normalize weights to fit in range 0, 1
        normalized = (weights - numpy.min(weights)) / (numpy.max(weights) - numpy.min(weights))
        # print(normalized)

        output_tokens = []
        output_weights = []

        assert isinstance(doc, TokenizedDocument)
        for i in range(len(normalized)):
            word = doc.tokens[i]
            weight = normalized[i]
            # print("%s [%.8f] " % (word, weight), end=' ')
            output_tokens.append(word)
            output_weights.append(float(weight))
        # print("\n------------")

        return output_tokens, output_weights


if __name__ == '__main__':
    # mInput = numpy.array([[1, 2, 3, 4]])
    # inShape = (4,)
    # net = Sequential()
    # outShape = 10
    # l1 = MyLayer(outShape, input_shape=inShape)
    # net.add(l1)
    # net.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # p = net.predict(x=mInput, batch_size=1)
    # print(p)

    my_input = numpy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
    net = Sequential()
    net.add(TimeDistributed(Dense(3)))
    # net.add(Dense(units=4, batch_input_shape=(None, 4, 3)))
    net.compile(optimizer='adam', loss='mean_absolute_error')
    net.predict(my_input)
    # model = Model(inputs=[my_input], outputs=)
