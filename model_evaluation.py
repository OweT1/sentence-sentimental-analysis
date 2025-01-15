import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def get_model_evaluation(y_true, y_pred, labels):
    confusionMatrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=labels)
    cm_display.plot()
    plt.show()

    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=labels))
    print(f"Accuracy: {accuracy_score(y_true=y_true, y_pred=y_pred)}" )
    print(f"Recall: {recall_score(y_true=y_true, y_pred=y_pred)}" )
    print(f"Precision: {precision_score(y_true=y_true, y_pred=y_pred)}" )
    print(f"F1-score: {f1_score(y_true=y_true, y_pred=y_pred)}" )