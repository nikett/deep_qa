# pylint: disable=no-self-use,invalid-name
from deep_qa.data import DataIndexer

from deep_qa.data.instances.text_classification.frame_instance import FrameInstance, IndexedFrameInstance
from tests.common.test_case import DeepQaTestCase


class TestFrameInstance(DeepQaTestCase):

    def setUp(self):
        super(TestFrameInstance, self).setUp()
        # Example of a typical input
        self.line = "event:plant absorb water###" \
                    "participant:water###" \
                    "agent:plant###" \
                    "finalloc:soil" \
                    + "\t" + "finalloc:soil"
        self.line_with_no_label_val = "event:plant absorb water###" \
                                      "participant:water###" \
                                      "agent:plant###" \
                                      "finalloc:soil" \
                                      + "\t" + "finalloc"
        self.padded_slots = ['plant', 'missingval', 'missingval', 'missingval', 'missingval',
                             'plant absorb water', 'ques', 'missingval', 'missingval',
                             'missingval', 'missingval', 'missingval', 'missingval', 'missingval', 'missingval',
                             'missingval', 'missingval', 'missingval', 'missingval', 'missingval', 'missingval',
                             'missingval', 'missingval', 'missingval', 'missingval', 'missingval', 'water']
        self.data_indexer = DataIndexer()
        for word in ['plant', 'missingval', 'absorb', 'ques', 'water', 'soil']:
            self.data_indexer.add_word_to_index(word)

    def test_convert_instance_to_indexed_instance(self):
        instance = FrameInstance.read_from_line(self.line)
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        assert indexed_instance.label == [self.data_indexer.get_word_index('soil')]

    def test_convert_instance_no_label_value_to_indexed_instance(self):
        instance = FrameInstance.read_from_line(self.line_with_no_label_val)
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        assert indexed_instance.label == [self.data_indexer.get_word_index('soil')]

    def test_slots_unwrap_correctly(self):
        instance = FrameInstance.read_from_line(self.line)
        # what we construct
        machine_label = instance.label
        machine_slot_values = instance.text
        # what we expect
        expected_label = "soil"
        # do they match?
        assert machine_label == expected_label
        assert machine_slot_values == self.padded_slots

    def test_slots_no_label_value_unwrap_correctly(self):
        instance = FrameInstance.read_from_line(self.line_with_no_label_val)
        # what we construct
        machine_label = instance.label
        machine_slot_values = instance.text
        # what we expect
        expected_label = "soil"
        # do they match?
        assert machine_label == expected_label
        assert machine_slot_values == self.padded_slots

    def test_words_from_frame_aggregated_correctly(self):
        instance = FrameInstance.read_from_line(self.line)
        assert len(instance.words()['words']) == 30

    def test_words_from_no_label_value_frame_aggregated_correctly(self):
        instance = FrameInstance.read_from_line(self.line_with_no_label_val)
        assert len(instance.words()['words']) == 30


class TestIndexedFrameInstance(DeepQaTestCase):

    def test_words_from_frame_aggregated_correctly(self):
        indexed_instance = IndexedFrameInstance([[1000], [1, 2, 3, 4, 5, 6, 7, 8],
                                                 [1, 2, 3]], [1, 2, 3])
        # unpadded label should be read correctly.
        assert indexed_instance.label == [1, 2, 3]
        padding_lengths = indexed_instance.get_padding_lengths()
        assert padding_lengths['num_sentence_words'] == 8
        indexed_instance.pad(padding_lengths)
        assert indexed_instance.label == [1, 2, 3, 0, 0, 0, 0, 0]
        assert indexed_instance.word_indices == [[1000, 0, 0, 0, 0, 0, 0, 0],
                                                 [1, 2, 3, 4, 5, 6, 7, 8],
                                                 [1, 2, 3, 0, 0, 0, 0, 0]]
