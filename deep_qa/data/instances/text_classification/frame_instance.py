from typing import Dict, List

import numpy
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


# Functionality supported in this class.
#
#    raw input line:
#            "event:plant absorb water###participant:water###agent:plant###finalloc:soil" TAB "agent:plant"
#
#    formatted to:
#           ["plant", "unk", "unk", "unk", "unk", "plant absorb water",
#            "soil", "unk", "unk", "unk", "unk", "unk",
#            "unk", "unk", "unk", "unk", "unk", "unk",
#            "unk", "unk", "unk", "unk", "unk", "unk",
#            "unk", "unk", "water"]
#
#    padded to a fixed length of say, 5 words per phrase.
#           ["plant P P P P",
#            "unk P P P P",
#            "unk P P P P",
#            "unk P P P P",
#            "unk P P P P",
#            "plant absorb water P P ",
#             ...]

# TODO PR request for having these in the json as an application specific content
# the slotnames can vary according to different end applications, e.g., a HowTo tuple, OpenIE tuple ...
SLOTNAMES_ORDERED = ["agent", "beneficiary", "causer", "context", "definition", "event",
                     "finalloc", "headverb", "initloc", "input", "output", "manner",
                     "patient", "resultant", "timebegin", "timeend", "temporal", "hierarchical",
                     "similar", "contemporary", "enables", "mechanism", "condition", "purpose",
                     "cause", "openrel", "participant"]
UNKNOWN_SLOTVAL = "unk"  # making an open world assumption, we do not observe all the values
QUES_SLOTVAL = "ques"  # this slot in the frame must be queried/completed.
MAX_PHRASE_LEN = 6

class FrameInstance(TextInstance):

    """
    A FrameInstance is a kind of TextInstance that has text in multiple slots. This generalizes a FrameInstance.
    """
    def __init__(self,
                 # the input can have only few slots and not all and in any random order.
                 # e.g., "event:plant absorb water###participant:water###agent:plant###finalloc:soil"
                 # TAB "agent:plant"
                 text: List[str],
                 # output is a phrase
                 label: str=None):
        super(FrameInstance, self).__init__(label)
        self.text = text  # "event:plant absorb water###participant:water###agent:plant" TAB "agent:plant"

    def __str__(self):
        return 'FrameInstance( [' + ',\n'.join(self.text) + '] , ' + str(self.label) + ')'

    @overrides
    # accumulate words from each slot's phrase
    def words(self) -> Dict[str, List[str]]:
        words = []
        for phrase in self.text:  # phrases
            phrase_words = self._words_from_text(phrase)
            words.extend(phrase_words['words'])
        label_words = self._words_from_text(self.label)
        words.extend(label_words['words'])
        return {"words": words}

    @staticmethod
    # slot_as_string: "participant:water" => name=participant, val=water
    def slot_from(slot_as_string: str, kv_separator: str=":"):
        slot_name_val = slot_as_string.split(kv_separator)
        return {'name': slot_name_val[0], 'val': slot_name_val[1]}

    @staticmethod
    # frame_as_string: "event:plant absorb water###participant:water" TAB "participant:water"
    def unpack_input(frame_as_string: str, kv_separator: str="\t"):
        # no information is lost in lowercasing, and simplifies matching.
        partialframe_query = frame_as_string.lower().split(kv_separator)
        if len(partialframe_query) != 2:
            raise RuntimeError("Unexpected number (not 2) of fields in frame: " + frame_as_string)
        return {'content': partialframe_query[0], 'query': partialframe_query[1]}

    @staticmethod
    def given_slots_from(slots_csv: str, values_separator: str="###", kv_separator: str=":"):
        return dict(map(lambda x: x.split(kv_separator), slots_csv.split(values_separator)))

    # Performs two types of padding:
    # i) unobserved slots are filled with self.unknown_slotval
    # ii) query slot is masked with self.unknown_queryval
    @staticmethod
    def padded_slots_from(sparse_frame: Dict[str, str], query_slotname: str):
        slots = []
        for slotname in SLOTNAMES_ORDERED:
            if slotname == query_slotname:  # query hence masked
                slots.append(QUES_SLOTVAL)
            elif slotname in sparse_frame:  # observed hence as-is
                slots.append(sparse_frame[slotname])
            else:  # unobserved hence padded
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
        Provides ordering and sparseness flexibility to the input text.
        """
        # Extract the query slot name and expected value
        # e.g. from, participant:water, extract the expected slot value "water"
        unpacked_input = cls.unpack_input(line)
        query_slot = cls.slot_from(unpacked_input['query'])
        given_slots = cls.given_slots_from(unpacked_input['content'])
        padded_slots = cls.padded_slots_from(given_slots, query_slot['name'])
        return cls(padded_slots, label=query_slot['val'])

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        # a phrase in a slot, is converted from [list of words] to [list of wordids]
        # this is repeated for every slot, hence a list of list of wordids/integers
        indices_slotvals = [self._index_text(phrase, data_indexer) for phrase in self.text]
        # the label is a phrase, and is converted from [list of words] to [list of wordids]
        indices_label = self._index_text(self.label, data_indexer)
        return IndexedFrameInstance(indices_slotvals, indices_label)


class IndexedFrameInstance(IndexedInstance):
    # One list of ints make up a slotvalue because a slotvalue is a phrase,
    # and so every word of the phrase is identified with an int id.
    # max length of a phrase is 6, if it is lesser, then pad.
    def __init__(self, word_indices: List[List[int]], label):
        super(IndexedFrameInstance, self).__init__(label)
        self.word_indices = word_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedFrameInstance([], label=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # the parent class somehow understands the semantics of these dictionary keys.
        # would be better if these keys were more explicit.
        return {'num_sentence_words': MAX_PHRASE_LEN}

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        """
        Pads (or truncates) all slot values to the maxlen
        e.g,
         input (phrases corresponding to each, the number of slots is fixed.)
         ["1000", "-1", "-1", "-1", "-1", "1 2 3",..]
         output (padded phrases, as phrases are composed of variable number of words)
         ["1000 0 0 0 0", "-1 0 0 0 0", "-1 0 0 0 0", "-1 0 0 0 0", "-1 0 0 0 0", "1 2 3 0 0",..]
        """
        truncate_from_right = False
        self.word_indices = [self.pad_word_sequence(indices, padding_lengths, truncate_from_right)
                             for indices in self.word_indices]
        self.label = self.pad_word_sequence(self.label, padding_lengths, truncate_from_right)

    @overrides
    def as_training_data(self):
        frame_matrix = numpy.asarray(self.word_indices, dtype='int32')
        label_list = numpy.asarray(self.label, dtype='int32')
        return frame_matrix, label_list
