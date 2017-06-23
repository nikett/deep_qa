from keras import backend as K
from keras.engine import Layer, InputSpec
from overrides import overrides


class AveragedBOWEncoder(Layer):
    """
    An encoder that averages (like a BOWEncoder) over a particular dimension of the tensor. 
    e.g., for a 4D tensor, averages over the specified dimension (e.g., = 2) 
    which is not the embedding dimension. The use case is suppose every token in a sequence 
    can be decomposed into multiple words. Then, embedding for each word is averaged.
    """
    def __init__(self, averaging_over_dim=-2, num_dimensions=3, **kwargs):
        self.input_spec = [InputSpec(ndim=num_dimensions)]
        if averaging_over_dim < 0:
            self.averaging_over_dim = num_dimensions + averaging_over_dim
        else:
            self.averaging_over_dim = averaging_over_dim
        # For consistency of handling sentence encoders, we will often get passed this parameter.
        # We don't use it, but Layer will complain if it's there, so we get rid of it here.
        kwargs.pop('units', None)
        super(AveragedBOWEncoder, self).__init__(**kwargs)

    @overrides
    def compute_output_shape(self, input_shape):
        # Drop the nth dimension. (n = self.averaging_over_dim)
        #
        # e.g., drop the phrase_len dimension in shape(batch, num_slots, phrase_len, embedding)
        # implies, return input_shape[0], input_shape[1], input_shape[3]
        return tuple([x for (index, x) in enumerate(input_shape) if index != self.averaging_over_dim])

    @overrides
    def compute_mask(self, inputs, mask=None):
        # We need to override this method because Layer passes the input mask unchanged since this layer
        # supports masking. We don't want that. After the input is averaged, we can stop propagating
        # the mask.
        return None

    @overrides
    def call(self, inputs, mask=None):
        # pylint: disable=redefined-variable-type
        if mask is None:
            return K.mean(inputs, axis=self.averaging_over_dim)  # axis = 1 was default
        else:
            # Compute weights such that masked elements have zero weights and the remaining
            # weight is distributed equally among the unmasked elements.
            # Mask (batch_size, num_slots, num_words_in_slot, embedding)
            # has 0s for masked elements and 1s everywhere else.
            # Mask is of type int8. While Theano would automatically make weighted_mask below
            # of type float32 even if mask remains int8, Tensorflow would complain. Let's cast it
            # explicitly to remain compatible with tf.
            float_mask = K.cast(mask, 'float32')

            # Get nth (e.g., 3rd) dimension from the tensor (batch, num_slots, phrase_len, embedding)
            # Perform multiplication: input x mask

            # Expanding dims of the denominator to make it the same shape as the numerator,
            # epsilon added to avoid division by zero.
            # (samples, num_words)
            updated_dim_to_avg = self.averaging_over_dim

            if K.ndim(float_mask) < K.ndim(inputs):
                updated_dim_to_avg -= 1
            weighted_mask = float_mask / (K.sum(float_mask, axis=updated_dim_to_avg, keepdims=True) + K.epsilon())
            if K.ndim(weighted_mask) < K.ndim(inputs):
                weighted_mask = K.expand_dims(weighted_mask)
            print("Weighted mask = ", weighted_mask)
            averaged = K.sum(inputs * weighted_mask, axis=self.averaging_over_dim)  # (batch_size, num_slots)
            print("Averaged = ", averaged)
            return averaged
