# pylint: disable=no-self-use,invalid-name
import numpy
from keras.layers import Input, Embedding
from keras.models import Model
from keras.initializers import ones

from deep_qa.layers.encoders.AveragedBOWEncoder import AveragedBOWEncoder
from tests.common.test_case import DeepQaTestCase


class TestAveragedBOWEncoder(DeepQaTestCase):

    def setUp(self):
        super(TestAveragedBOWEncoder, self).setUp()
        # Example of a typical input
        self.dim_to_avg = 2
        self.num_axes = 3

    def test_on_unmasked_input(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding does not mask zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)
        encoder = AveragedBOWEncoder(self.dim_to_avg, self.num_axes)
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        expected_output = numpy.mean(embedding_weights[test_input], axis=self.dim_to_avg)
        actual_output = model.predict(test_input)
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_masked_input(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        encoder = AveragedBOWEncoder(self.dim_to_avg, self.num_axes)
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        # Omitting the first element (0), because that is supposed to be masked in the model.
        print("embedding weights", embedding_weights)
        expected_output = numpy.mean(embedding_weights[test_input[:, 1:]], axis=self.dim_to_avg)
        print("expected output: ", expected_output)
        actual_output = model.predict(test_input)
        print("actual output : ", actual_output)
        # Following comparison is till the sixth decimal.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_masked_ones(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 5  # 15 originally
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size,
                              output_dim=embedding_dim,
                              mask_zero=True,
                              embeddings_initializer=ones())
        encoder = AveragedBOWEncoder(self.dim_to_avg, self.num_axes)
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        print("embedded input", embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 2, 4]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        print("embedding weights", embedding_weights)
        ew_first_elem_omitted = embedding_weights[test_input[:, 1:]]
        print("embedding weights removing masked: ", ew_first_elem_omitted)
        # Omitting the first element (0), because that is supposed to be masked in the model.
        expected_output = numpy.mean(ew_first_elem_omitted, axis=self.dim_to_avg)  # self.dim_to_avg)
        print("expected output: ", expected_output)
        print("expected output shape: ", expected_output.shape)
        print("test input : ", test_input)

        actual_output = model.predict(test_input)
        print("actual output : ", actual_output)
        print("actual output shape: ", actual_output.shape)
        # Following comparison is till the sixth decimal.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_all_zeros(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        encoder = AveragedBOWEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 0, 0, 0, 0]], dtype='int32')
        # Omitting the first element (0), because that is supposed to be masked in the model.
        expected_output = numpy.zeros((1, embedding_dim))
        actual_output = model.predict(test_input)
        # Following comparison is till the sixth decimal.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_unmasked_input_orig(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding does not mask zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)
        encoder = AveragedBOWEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        expected_output = numpy.mean(embedding_weights[test_input], axis=1)
        actual_output = model.predict(test_input)
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_masked_input_orig(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        encoder = AveragedBOWEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        # Omitting the first element (0), because that is supposed to be masked in the model.
        expected_output = numpy.mean(embedding_weights[test_input[:, 1:]], axis=1)
        print(expected_output.shape)
        actual_output = model.predict(test_input)
        print(actual_output.shape)
        # Following comparison is till the sixth decimal.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_all_zeros_orig(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        encoder = AveragedBOWEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 0, 0, 0, 0]], dtype='int32')
        # Omitting the first element (0), because that is supposed to be masked in the model.
        expected_output = numpy.zeros((1, embedding_dim))
        actual_output = model.predict(test_input)
        # Following comparison is till the sixth decimal.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)
