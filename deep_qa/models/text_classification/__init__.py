from .framecloze_model import FrameClozeModel
from .classification_model import ClassificationModel

concrete_models = {  # pylint: disable=invalid-name
        'ClassificationModel': ClassificationModel,
        'FrameClozeModel': FrameClozeModel,
        }
