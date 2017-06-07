from typing import Dict, List

import numpy
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer

# TODO PR request for having these in the json as an application specific content
# the slotnames can vary according to different end applications, e.g., a HowTo tuple, OpenIE tuple ...
SLOTNAMES_ORDERED = ["agent", "beneficiary", "causer", "context", "definition", "event",
                     "finalloc", "headverb", "initloc", "input", "output", "manner",
                     "patient", "resultant", "timebegin", "timeend", "temporal", "hierarchical",
                     "similar", "contemporary", "enables", "mechanism", "condition", "purpose",
                     "cause", "openrel", "participant"]
UNKNOWN_SLOTVAL = "unk"  # making an open world assumption, we do not observe all the values
QUES_SLOTVAL = "ques"  # this slot in the frame must be queried/completed.


class FrameInstance(TextInstance):

    """
    A FrameInstance is a kind of TextInstance that has text in multiple slots. This generalizes a FrameInstance.
    """
    def __init__(self,
                 dense_frame: List[str],
                 phrase_in_queried_slot: str=None):  # output label is a phrase
        super(FrameInstance, self).__init__(phrase_in_queried_slot)
        self.text = dense_frame  # "event:plant absorb water###participant:water###agent:plant" TAB "agent:plant"

    def __str__(self):
        return 'FrameInstance( [' + ',\n'.join(self.text) + '] , ' + str(self.label) + ')'

    @overrides
    def words(self) -> Dict[str, List[str]]:
        # Accumulate words from each slot's phrase.
        # Label is also a phrase, so additionally accumulate words from label
        words = []
        for phrase in self.text:  # phrases
            phrase_words = self._words_from_text(phrase)
            words.extend(phrase_words['words'])
        label_words = self._words_from_text(self.label)
        words.extend(label_words['words'])
        return {"words": words}

    @staticmethod
    def query_slot_from(slot_as_string: str,
                        sparse_given_frame: Dict[str, str],
                        kv_separator: str=":"):
        """
        :param slot_as_string: "participant:water"
        :param sparse_given_frame: If the expected slot name is given in the query
        but its value is not, then pick the value from the sparse_given_frame
        :param kv_separator: typically colon separated
        :return: name=participant, val=water
        """
        slot_name_val = slot_as_string.split(kv_separator)
        # Suppose slot_as_string is: participant (i.e. no value is specified)
        # this is assumed as participant:BLANK_VALUE, if we cannot look it up in the partial frame.
        if len(slot_name_val) == 1:
            slot_name_val = (slot_as_string + ":" +
                             sparse_given_frame.get(slot_name_val[0], '')).split(kv_separator)
        return {'name': slot_name_val[0], 'val': slot_name_val[1]}

    @staticmethod
    def unpack_input(frame_as_string: str,
                     kv_separator: str="\t"):
        """
        :param frame_as_string: "event:plant absorb water###participant:water" TAB "participant:water"
        :param kv_separator: typically TAB separated partial frame and query
        :return: event:plant absorb water###participant:water, and query: participant:water
                Both event and query will be lowercased
        """
        # No information loss in lower-casing, and simplifies matching.
        partialframe_query = frame_as_string.lower().split(kv_separator)
        if len(partialframe_query) != 2:
            raise RuntimeError("Unexpected number (not 2) of fields in frame: " + frame_as_string)
        return {'content': partialframe_query[0], 'query': partialframe_query[1]}

    @staticmethod
    def given_slots_from(slots_csv: str,
                         values_separator: str="###",
                         kv_separator: str=":"):
        """
        :param slots_csv: event:plant absorb water###participant:water
        :param values_separator: typically "###"
        :param kv_separator: typically ":"
        :return: map of slotnames -> slot phrase [event -> plant absorb water , participant -> water]
        """
        return dict(map(lambda x: x.split(kv_separator), slots_csv.split(values_separator)))

    @staticmethod
    def dense_frame_from(sparse_frame: Dict[str, str],
                         query_slotname: str):
        """
        Performs two types of padding:
        i) unobserved slots are filled with self.unknown_slotval
        ii) query slot is masked with self.unknown_queryval
        The order of slots strictly follows from SLOTNAMES_ORDERED.
        :param sparse_frame:
                slotnames -> slot phrase [event -> plant absorb water , participant -> water]
        :param query_slotname:
                participant
        :return: [plant absorb water, ques, unk, unk, ...]
        """
        slots = []
        for slotname in SLOTNAMES_ORDERED:
            if slotname == query_slotname:  # query hence masked
                slots.append(QUES_SLOTVAL)
            elif slotname in sparse_frame:  # observed hence as-is
                slots.append(sparse_frame[slotname])
            else:  # unobserved hence inserted
                slots.append(UNKNOWN_SLOTVAL)
        return slots

    @classmethod
    @overrides
    def read_from_line(cls, line: str):
        """
        Reads a FrameInstance from a line.  The format is:
        frame represented as list of <role:role value phrase of maxlen 5> TAB <label>
        e.g., from
        event:plant absorb water###participant:water###agent:plant###finalloc:soil
              to
        ["plant", "unk", "unk", "unk", "unk", "plant absorb water",
          "soil", "unk", "unk", "unk", "unk", "unk",
          "unk", "unk", "unk", "unk", "unk", "unk",
          "unk", "unk", "unk", "unk", "unk", "unk",
          "unk", "unk", "water"]
        Provides ordering (input can be composed of slots in arbitrary order)
        and sparseness flexibility (only a few slots can be mentioned in the input).
        """
        # Extract the query slot name and expected value
        # e.g. from, participant:water, extract the expected slot value "water"
        unpacked_input = cls.unpack_input(line)
        given_sparse_frame = cls.given_slots_from(unpacked_input['content'])
        query_slot = cls.query_slot_from(unpacked_input['query'], given_sparse_frame)
        dense_frame = cls.dense_frame_from(given_sparse_frame, query_slot['name'])
        return cls(dense_frame, phrase_in_queried_slot=query_slot['val'])

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        # A phrase in a slot, is converted from list of words to list of wordids.
        # This is repeated for every slot, hence a list of list of wordids/integers.
        indices_slotvals = [self._index_text(phrase, data_indexer) for phrase in self.text]
        # The label is a phrase, and is converted from list of words to list of wordids.
        indices_label = self._index_text(self.label, data_indexer)
        return IndexedFrameInstance(indices_slotvals, indices_label)


class IndexedFrameInstance(IndexedInstance):
    """
    Ensures that a phrase in every slot, and the label (also a phrase) are padded to be of a fixed maxlen.
    Max length of a phrase is 6 (configurable), pad phrases with fewer words; if it exceeds 6 then truncate.
    """
    def __init__(self, word_indices: List[List[int]], label):
        """
        :param word_indices: One list of ints make up a slotvalue because a slotvalue is a phrase,
                             and so every word of the phrase is identified with an int id.
        :param label: phrase, hence a list of ints.
        """
        super(IndexedFrameInstance, self).__init__(label)
        self.word_indices = word_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedFrameInstance([], label=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # Record the length of every slot content
        # Let the model pad to the max phrase length#
        # e.g., ["1000", "1", "1", "1", "1", "1 2 3",..]
        # slotlen  1,1,1,1,3..
        # Expected: Dict ['some key', 3] which is the max phrase len across all slots.
        all_slot_lengths = [self._get_word_sequence_lengths(slot_indices) for slot_indices in self.word_indices]
        # find the max from all_slot_lengths
        lengths = {}
        for key in all_slot_lengths[0]:
            lengths[key] = max(slot_lengths[key] for slot_lengths in all_slot_lengths)
        return lengths

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        """
        Pads (or truncates) all slot values to the maxlen
        Input: (phrases corresponding to each slot, the number of slots is fixed.)
        e.g., ["1000", "1", "1", "1", "1", "1 2 3",..]
        Note: these are arrays over phrase word ids.
        Output: (padded phrases, as phrases are composed of variable number of words)
        e.g., ["1000 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 2 3 0 0",..]
        Note: padding is fixed length, anything larger or small will be pruned. Phrases are truncated from left.
        """
        truncate_from_right = False
        self.word_indices = [self.pad_word_sequence(indices, padding_lengths, truncate_from_right)
                             for indices in self.word_indices]
        self.label = self.pad_word_sequence(self.label, padding_lengths, truncate_from_right)

    @overrides
    def as_training_data(self):
        # The frame and the label must be numpy matrix and array respectively
        frame_as_matrix = numpy.asarray(self.word_indices, dtype='int32')
        label_as_list = numpy.asarray(self.label, dtype='int32')
        return frame_as_matrix, label_as_list
