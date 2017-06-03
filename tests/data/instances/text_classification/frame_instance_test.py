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
                    "finalloc:soil"\
                    + "\t" + "finalloc:soil"
        self.padded_slots = ['plant', 'unk', 'unk', 'unk', 'unk',
                             'plant absorb water', 'ques', 'unk', 'unk',
                             'unk', 'unk', 'unk', 'unk', 'unk', 'unk',
                             'unk', 'unk', 'unk', 'unk', 'unk', 'unk', 'unk',
                             'unk', 'unk', 'unk', 'unk', 'water']
        self.data_indexer = DataIndexer()
        for word in ['plant', 'unk', 'absorb', 'ques', 'water', 'soil']:
            self.data_indexer.add_word_to_index(word)

    def tearDown(self):
        super(TestFrameInstance, self).tearDown()

    def test_convert_instance_to_indexed_instance(self):
        instance = FrameInstance.read_from_line(self.line)
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        print(instance.label + "\t padded to: " + indexed_instance.label.__str__())
        print(indexed_instance.word_indices.__str__())
        assert indexed_instance.label == [self.data_indexer.get_word_index('soil')]  # TODO

    def test_slots_unwrap_correctly(self):
        instance = FrameInstance.read_from_line(self.line)
        # what we get
        # TODO: consider when label is OOV or finalloc instead of finalloc:soil
        machine_label = instance.label
        machine_slotvals = instance.text
        # what we expect
        expected_label = "soil"
        # do they match?
        assert machine_label == expected_label
        assert machine_slotvals == self.padded_slots

    def test_words_from_frame_aggregated_correctly(self):
        instance = FrameInstance.read_from_line(self.line)
        assert len(instance.words()['words']) == 30


class TestIndexedFrameInstance(DeepQaTestCase):

    def test_words_from_frame_aggregated_correctly(self):
        indexed_instance = IndexedFrameInstance([[1000], [1, 2, 3, 4, 5, 6, 7, 8],
                                                 [1, 2, 3]], [1, 2, 3])
        # unpadded label should be read correctly.
        assert indexed_instance.label == [1, 2, 3]
        padding_lengths = indexed_instance.get_padding_lengths()
        assert padding_lengths['num_sentence_words'] == 6
        indexed_instance.pad(padding_lengths)
        assert indexed_instance.label == [1, 2, 3, 0, 0, 0]
        assert indexed_instance.word_indices == [[1000, 0, 0, 0, 0, 0],
                                                 [1, 2, 3, 4, 5, 6],
                                                 [1, 2, 3, 0, 0, 0]]
