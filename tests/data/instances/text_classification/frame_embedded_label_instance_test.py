# pylint: disable=no-self-use,invalid-name
import numpy

from deep_qa.data import DataIndexer
from deep_qa.data.instances.text_classification.frame_embedded_label_instance import FrameEmbeddedLabelInstance, \
    IndexedNumericalFrameInstance
from tests.common.test_case import DeepQaTestCase


class TestFrameEmbeddedLabelInstance(DeepQaTestCase):

    embedded_label_str = "0.0891758,0.121832,-0.0671959,0.0477279,-0.013659,-0.0671959,0.0640559,-0.0331269 , " \
                              "-0.0364239,0.00565199,-0.017113,-0.10362,0.0552639,-0.00706499,-0.0643699,0.08," \
                              "0.110528,-0.108644,0.00374837,-0.020567,-0.0464719,-0.015386,0.0979678,-0.02364," \
                              "-0.012717,0.0251199,-0.0389359,0.0828958,0.10676,0.0390929,0.0756738,0.0140515," \
                              "-0.021823,0.162024,0.0941998,-0.0118535,-0.0452159,-0.0298299,0.0423899,0.0712," \
                              "0.002487,-0.00883123,0.0577759,-0.0189185,0.0168775,0.0408199,-0.0405059,0.0539," \
                              "0.0891758,0.121832,-0.0671959,0.0477279,-0.013659,-0.069,0.0640559,-0.0331269," \
                              "-0.0364239,0.00565199,-0.017113,-0.10362,0.0552639,-0.00706499,-0.0643699,0.08," \
                              "0.110528,-0.108644,0.00374837,-0.020567,-0.0464719,-0.015386,0.0979678,-0.02364," \
                              "-0.012717,0.0251199,-0.0389359,0.0828958,0.10676,0.0390929,0.0756738,0.0140515," \
                              "-0.021823,0.162024,0.0941998,-0.0118535,-0.0452159,-0.0298299,0.0423899,0.0712," \
                              "0.002487,-0.00883123,0.0577759,-0.0189185,0.0168775,0.0408199,-0.0405059,0.0539," \
                              "-0.0480419,-0.0277889,0.0872918,-0.0189185"

    embedded_label = numpy.array(list(embedded_label_str.replace(' ', '').split(',')), dtype='float64')

    def setUp(self):
        super(TestFrameEmbeddedLabelInstance, self).setUp()
        # Example of a typical input
        self.line = "event:plant absorb water###" \
                    "participant:water###" \
                    "agent:plant###" \
                    "finalloc:" \
                    + "\t" + "finalloc:"+self.embedded_label_str
        self.line_with_no_label_val = "event:plant absorb water###" \
                                      "participant:water###" \
                                      "agent:plant" \
                                      + "\t" + "finalloc:"+self.embedded_label_str
        self.padded_slots = ['plant', 'missingval', 'missingval', 'missingval', 'missingval',
                             'plant absorb water', 'ques', 'missingval', 'missingval',
                             'missingval', 'missingval', 'missingval', 'missingval', 'missingval', 'missingval',
                             'missingval', 'missingval', 'missingval', 'missingval', 'missingval', 'missingval',
                             'missingval', 'missingval', 'missingval', 'missingval', 'missingval', 'water']
        self.data_indexer = DataIndexer()
        for word in ['plant', 'missingval', 'absorb', 'ques', 'water']:
            self.data_indexer.add_word_to_index(word)

    def test_convert_instance_to_indexed_instance(self):
        instance = FrameEmbeddedLabelInstance.read_from_line(self.line)
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        assert (indexed_instance.label == self.embedded_label).all()

    def test_slots_unwrap_correctly(self):
        instance = FrameEmbeddedLabelInstance.read_from_line(self.line)
        # what we construct
        machine_label = instance.label
        machine_slot_values = instance.text
        # do they match?
        assert (machine_label == self.embedded_label).all()
        assert machine_slot_values == self.padded_slots

    def test_words_from_frame_aggregated_correctly(self):
        instance = FrameEmbeddedLabelInstance.read_from_line(self.line)
        # Compared to analogous test in FrameInstanceTest, "soil" is missing as a true label.
        assert len(instance.words()['words']) == 29

    def test_words_from_no_label_value_frame_aggregated_correctly(self):
        instance = FrameEmbeddedLabelInstance.read_from_line(self.line_with_no_label_val)
        assert len(instance.words()['words']) == 29


class TestIndexedFrameInstance(DeepQaTestCase):
    embedded_label_str = "0.0891758,0.121832,-0.0671959,0.0477279,-0.013659,-0.0671959,0.0640559,-0.0331269 , " \
                              "-0.0364239,0.00565199,-0.017113,-0.10362,0.0552639,-0.00706499,-0.0643699,0.08," \
                              "0.110528,-0.108644,0.00374837,-0.020567,-0.0464719,-0.015386,0.0979678,-0.02364," \
                              "-0.012717,0.0251199,-0.0389359,0.0828958,0.10676,0.0390929,0.0756738,0.0140515," \
                              "-0.021823,0.162024,0.0941998,-0.0118535,-0.0452159,-0.0298299,0.0423899,0.0712," \
                              "0.002487,-0.00883123,0.0577759,-0.0189185,0.0168775,0.0408199,-0.0405059,0.0539," \
                              "0.0891758,0.121832,-0.0671959,0.0477279,-0.013659,-0.069,0.0640559,-0.0331269," \
                              "-0.0364239,0.00565199,-0.017113,-0.10362,0.0552639,-0.00706499,-0.0643699,0.08," \
                              "0.110528,-0.108644,0.00374837,-0.020567,-0.0464719,-0.015386,0.0979678,-0.02364," \
                              "-0.012717,0.0251199,-0.0389359,0.0828958,0.10676,0.0390929,0.0756738,0.0140515," \
                              "-0.021823,0.162024,0.0941998,-0.0118535,-0.0452159,-0.0298299,0.0423899,0.0712," \
                              "0.002487,-0.00883123,0.0577759,-0.0189185,0.0168775,0.0408199,-0.0405059,0.0539," \
                              "-0.0480419,-0.0277889,0.0872918,-0.0189185"
    embedded_label = numpy.array(list(embedded_label_str.replace(' ', '').split(',')), dtype='float64')

    def test_words_from_frame_aggregated_correctly(self):
        # word ids in phrases, that would be eventually padded
        indexed_instance = IndexedNumericalFrameInstance([[1000], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3]],
                                                         self.embedded_label)
        # unpadded label should be read correctly.
        assert (indexed_instance.label == TestFrameEmbeddedLabelInstance.embedded_label).any()
        padding_lengths = indexed_instance.get_padding_lengths()
        assert padding_lengths['num_sentence_words'] == 8
        indexed_instance.pad(padding_lengths)
        assert (indexed_instance.label == TestFrameEmbeddedLabelInstance.embedded_label).any()
        assert indexed_instance.word_indices == [[1000, 0, 0, 0, 0, 0, 0, 0],
                                                 [1, 2, 3, 4, 5, 6, 7, 8],
                                                 [1, 2, 3, 0, 0, 0, 0, 0]]
