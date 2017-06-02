# pylint: disable=no-self-use,invalid-name

from deep_qa.data.instances.text_classification.frame_instance import FrameInstance, IndexedFrameInstance

# Example of a typical input
line = "event:plant absorb water###participant:water###agent:plant###finalloc:soil"+"\t"+"finalloc:soil"


class TestFrameInstance:

    # TODO more test with boundary conditions on imperfect input.
    def test_slots_unwrap_correctly(self):
        instance = FrameInstance.read_from_line(line)
        # what we get
        machine_label = "" + instance.label
        machine_slotvals = ""+instance.text.__str__()
        # what we expect
        expected_label = "soil"
        expected_slotsvals = "['plant', 'unk', 'unk', 'unk', 'unk', " \
                             "'plant absorb water', 'ques', 'unk', 'unk', " \
                             "'unk', 'unk', 'unk', 'unk', 'unk', 'unk', " \
                             "'unk', 'unk', 'unk', 'unk', 'unk', 'unk', 'unk', " \
                             "'unk', 'unk', 'unk', 'unk', 'water']"
        # do they match?
        assert machine_label == expected_label
        assert machine_slotvals == expected_slotsvals

    def test_words_from_frame_aggregated_correctly(self):
        instance = FrameInstance.read_from_line(line)
        assert instance.words()['words'].__len__() == 30


class TestIndexedFrameInstance:

    def test_words_from_frame_aggregated_correctly(self):
        # TODO open a PR request to provide fixed length padding e.g. 6
        indexed_instance = IndexedFrameInstance([[1000], [1, 2, 3, 4, 5, 6, 7, 8],
                                                 [1, 2, 3]], [1, 2, 3])
        # unpadded label should be read correctly.
        assert indexed_instance.label.__str__() == "[1, 2, 3]"
        padding_lengths = indexed_instance.get_padding_lengths()
        assert padding_lengths.__str__() == "{'num_sentence_words': 6}"
        indexed_instance.pad(padding_lengths)
        assert indexed_instance.label.__str__() == "[1, 2, 3, 0, 0, 0]"
        assert indexed_instance.word_indices.__str__() == "[[1000, 0, 0, 0, 0, 0], " \
                                                          "[1, 2, 3, 4, 5, 6], " \
                                                          "[1, 2, 3, 0, 0, 0]]"
