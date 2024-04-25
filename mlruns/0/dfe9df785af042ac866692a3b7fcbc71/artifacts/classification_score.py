import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import mlflow


# ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig('roc_curve.png')
    plt.close()

# PR curve
def plot_pr_curve(precision, recall, average_precision):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')   
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
    plt.savefig('pr_curve.png')
    plt.close()

def confusion_matrix_plot(clf, X_val, y_val):

    ConfusionMatrixDisplay.from_estimator(clf, X_val, y_val)
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('confusion_matrix.png')
    plt.close() 

    

        
        
# Classification score
def clf_score(clf, X_train, y_train, X_val, y_val, train=True):
    if train:
        print("Train Result:\n")
        train_accuracy = accuracy_score(y_train, clf.predict(X_train))
        print("accuracy score: {0:.4f}\n".format(train_accuracy))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_average_accuracy", np.mean(res))
        mlflow.log_metric("train_accuracy_sd", np.std(res))
        mlflow.log_param("cross_validation_folds", 10)  
        
    elif train == False:
        print("Validation Result:\n")
        validation_accuracy = accuracy_score(y_val, clf.predict(X_val))
        print("accuracy score: {0:.4f}\n".format(validation_accuracy))
        
        precision, recall, _ = precision_recall_curve(y_val, clf.predict(X_val))
        average_precision = average_precision_score(y_val, clf.predict(X_val))
        plot_pr_curve(precision, recall, average_precision)
        
        
        fpr, tpr, _ = roc_curve(y_val, clf.predict(X_val))
        roc_auc = roc_auc_score(y_val, clf.predict(X_val))
        print("roc auc score: {}\n".format(roc_auc))
        plot_roc_curve(fpr, tpr, roc_auc)

        
        
        print("Classification Report: \n {}\n".format(classification_report(y_val, clf.predict(X_val))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_val, clf.predict(X_val))))
        confusion_matrix_plot(clf, X_val, y_val)
 
        print("End of validation Result\n")
        mlflow.log_metric("validation_accuracy", validation_accuracy)
        mlflow.log_metric("average_precision", average_precision)
        mlflow.log_metric("roc_auc", roc_auc)
        
        
