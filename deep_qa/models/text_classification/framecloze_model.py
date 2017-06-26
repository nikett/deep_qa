from overrides import overrides

from keras.layers import Dense, Dropout, Input

# from deep_qa.data.instances.text_classification import concrete_instances
from deep_qa.data.instances.sequence_tagging import concrete_instances
from deep_qa.layers.encoders import BOWEncoder
from deep_qa.layers.encoders.AveragedBOWEncoder import AveragedBOWEncoder
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel
from ...common.params import Params


class FrameClozeModel(TextTrainer):
    """
    This ``FrameClozeModel`` is a type of text train with:
    ------------
    Input/ Output
    ------------
    input: a partial frame (list of phrases in a fixed order), a query (slot to complete).
    predicts: an embedding for the query slot
    where,
    - partial frame is a list of phrases in a fixed order, see ``FrameInstance`` for examples,
    - query is the slot to complete. Instead of a one hot vector we provide it with a default token (ques)
    - embedding for the queried slot is the BOW representation of the phrase value of the queried slot.
    ---------------------
    Network architecture
    ---------------------
    We use BOW encoder along with a dense layer.
    We use stacked seq2seq encoders followed by a dense layer.
    ----------
    Parameters
    ----------
    # num_stacked_rnns : int, optional (default: ``1``)
        The number of ``seq2seq_encoders`` that we should stack on top of each other before
        predicting tags.
    # instance_type : str
        Specifies the instance type, currently the only supported type is "FrameInstance",
        which defines things like how the input data is formatted and tokenized.
    """
    def __init__(self, params: Params):
        self.num_stacked_rnns = params.pop('num_stacked_rnns', 1)
        instance_type = params.pop('instance_type', "FrameInstance")
        self.nearest_neighbor_dim = params.pop('nearest_neighbor_dim', 200)
        self.instance_type = concrete_instances[instance_type]  # TODO is this a bug?
        super(FrameClozeModel, self).__init__(params)
        self.num_slots = self._instance_type().words['slot_names']

    @overrides
    def _build_model(self):

        # Input: (slots, query-slot)
        # Output: (queried-slot-phrase-embedding)

        # Step 1: Convert the sentence input into sequences of word vectors.

        # train_input: numpy array: int32 (batch_size, num_slots, text_length).
        # Left padded arrays of word indices from sentences in training data.
        # We have a list of phrases as input. The base class implementation of
        # _get_sentence_shape provides sentence length, which is the phrase length
        # in this model. We shall additionally supply number of slots.
        slots_input = Input(shape=(self._get_sentence_shape(),
                                   self.num_slots),
                            dtype='int32',
                            name="slots_input")

        # Step 2: Pass the sequences of word vectors through the sentence encoder to get a sentence vector.
        # Shape: (batch_size, number_of_slots, max_phrase_len, embedding_dim)
        each_slot_embedding = self._embed_input(slots_input)

        # average out over phrase_len:
        # from: batch_size, number_of_slots, phrase_len, embedding_dim
        # output should become: batch_size, number_of_slots, embedding_dim
        averaging_layer = AveragedBOWEncoder(2, 4)
        # batch_size, number_of_slots, embedding_dim
        each_slot_embedding = averaging_layer(each_slot_embedding)

        # Shape: (batch_size, number_of_slots, embedding_dim)
        # We first convert a sentence to a sequence of word embeddings
        # and then apply a stack of seq2seq encoders.
        for i in range(self.num_stacked_rnns):
            encoder = self._get_seq2seq_encoder(name="encoder_{}".format(i),
                                                fallback_behavior="use default params")
            # shape still (batch_size, number_of_slots, 2 * embedding_dim)
            each_slot_embedding = encoder(each_slot_embedding)

        # From (batch_size, number_of_slots, 2 * embedding_dim),
        # convert to batch_size, 2*embedding_dim
        bow_features = BOWEncoder()
        avg_slot_embedding = bow_features(each_slot_embedding)
        # Add a dropout after LSTM.
        regularized_embedding = Dropout(0.2)(avg_slot_embedding)

        # Step 3: Dense projection
        # From:(batch_size, 2*embedding_dim),
        # convert to (batch_size, nn_embedding_dim),
        # so, a dense layer is needed
        projection_layer = Dense(int(self.nearest_neighbor_dim), activation='relu', name='projector')
        projected_frame = projection_layer(regularized_embedding)

        # Step 4: Define squared loss against labels as the loss.
        # TODO: this requires that training input contain a vector representation of the queried slot as label.
        # Further, we need to find all the possible nearest neighbors for this vector.
        return DeepQaModel(inputs=slots_input, outputs=projected_frame)

    def _instance_type(self):
        return self.instance_type

    @overrides
    def _set_padding_lengths_from_model(self):
        # We return the dimensions of
        # 0th layer which is "indexed input" (0),
        # 0th item in the input slot which is "phrase" [0],
        # and everything that comes after the batch_size which "includes #words, #characters etc." [1:]
        self._set_text_lengths_from_model_input(self.model.get_input_shape_at(0)[0][1:])

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(FrameClozeModel, cls)._get_custom_objects()
        # If we use any custom layers implemented in deep_qa (not part of original Keras),
        # they need to be added in the custom_objects dictionary.
        custom_objects["BOWEncoder"] = BOWEncoder
        return custom_objects
