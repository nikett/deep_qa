from deep_qa.data.instances.text_classification.frame_embedded_label_instance import FrameEmbeddedLabelInstance
from deep_qa.data.instances.text_classification.frame_instance import FrameInstance
from .logical_form_instance import LogicalFormInstance, IndexedLogicalFormInstance
from .text_classification_instance import TextClassificationInstance, IndexedTextClassificationInstance
from .tuple_instance import TupleInstance, IndexedTupleInstance

concrete_instances = {
        'FrameInstance': FrameInstance,
        'FrameEmbeddedLabelInstance': FrameEmbeddedLabelInstance
        }
