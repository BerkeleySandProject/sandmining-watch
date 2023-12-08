from .schemas import AnnotationType, AnnotationConfig

### Annotations

from rastervision.core.data import ClassConfig
# CLASS_NAME2 = 'sandmine'
CLASS_CONFIG2 = ClassConfig(
    colors=['grey', 'red'],
    names=['other', 'sandmine'],
    null_class='other'
) 

CLASS_CONFIG3 = ClassConfig(
    colors=['grey', 'yellow', 'red'],
    names=['other','lc', 'sandmine'],
    null_class='other'
) 



two_class_config = AnnotationConfig(AnnotationType.TWO_ClASS,
                                    class_config=CLASS_CONFIG2,
                                    num_classes=2,
                                    labelbox_project_id = "cllbeyixh0bxt07uxfvg977h3")

three_class_config = AnnotationConfig(AnnotationType.THREE_CLASS,
                                    class_config=CLASS_CONFIG3,
                                    num_classes=3,
                                    labelbox_project_id = "cloqcy8v201c507yb5ar341g2",
                                    postfix="_3class")  # needed when finding the GCP annotations path. by default 2_class doesnt need this