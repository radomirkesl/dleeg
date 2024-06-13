from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

CLASS_NUM = 4


def build_general_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "accuracy_macro": MulticlassAccuracy(
                num_classes=CLASS_NUM, average="macro"
            ),
            "accuracy_micro": MulticlassAccuracy(
                num_classes=CLASS_NUM, average="micro"
            ),
            "precision_macro": MulticlassPrecision(
                num_classes=CLASS_NUM, average="macro"
            ),
            "precision_micro": MulticlassPrecision(
                num_classes=CLASS_NUM, average="micro"
            ),
            "recall_macro": MulticlassRecall(num_classes=CLASS_NUM, average="macro"),
            "recall_micro": MulticlassRecall(num_classes=CLASS_NUM, average="micro"),
            "f1_macro": MulticlassF1Score(num_classes=CLASS_NUM, average="macro"),
            "f1_micro": MulticlassF1Score(num_classes=CLASS_NUM, average="micro"),
            "auroc_macro": MulticlassAUROC(num_classes=CLASS_NUM, average="macro"),
        }
    )


def _build_classwise_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "accuracy_classwise": MulticlassAccuracy(
                num_classes=CLASS_NUM, average=None
            ),
            "precision_classwise": MulticlassPrecision(
                num_classes=CLASS_NUM, average=None
            ),
            "recall_classwise": MulticlassRecall(num_classes=CLASS_NUM, average=None),
            "f1_classwise": MulticlassF1Score(num_classes=CLASS_NUM, average=None),
            "auroc_classwise": MulticlassAUROC(num_classes=CLASS_NUM, average=None),
        }
    )
