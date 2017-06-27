from deep_qa.common.params import Params
from deep_qa.models.text_classification import FrameClozeModel
from tests.common.test_case import DeepQaTestCase


class TestFrameClozeModel(DeepQaTestCase):
    def test_trains_and_loads_correctly(self):
        self.write_frame_cloze_files()
        pre_ = "/Users/nikett/Documents/work/code/thirdparty/deepqa/deep_qa/datasets-pulled/glove.6B.100d.txt.gz"
        args = Params({
                'save_models': True,
                'show_summary_with_masking_info': True,
                'instance_type': 'FrameEmbeddedLabelInstance',
                'validation_metric': 'val_loss',
                # 'model_serialization_prefix': '/Users/nikett/TEMP/',
                'loss': 'mean_squared_error',
                # 'num_slots': 27, TODO: why? ->  "Extra parameters passed to Trainer: {'num_slots': 27}"
                "embeddings": {"words":
                               {"dimension": 100,
                                "pretrained_embeddings_file": pre_
                               },
                               "characters": {"dimension": 8}
                              },
                'tokenizer': {'processor': {'word_splitter': 'simple'}},
                })
        self.ensure_model_trains_and_loads(FrameClozeModel, args)
