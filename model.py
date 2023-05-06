from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from collections import defaultdict
import argparse
import os
import json

def load_train_data(problem_folder):
    author_folders = [f for f in os.listdir(problem_folder) if f.startswith("candidate")]
    X = []
    y = []

    for author_folder in author_folders:
        author_path = os.path.join(problem_folder, author_folder)
        author_files = os.listdir(author_path)

        for author_file in author_files:
            file_path = os.path.join(author_path, author_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                file = f.read()
            X.append(file)
            y.append(author_folder)

    return X, y

def load_test_data(problem_folder):
    unknown_path = os.path.join(problem_folder, 'unknown')
    unknown_files = os.listdir(unknown_path)
    X = []
    y = []

    for unknown_file in unknown_files:
        file_path = os.path.join(unknown_path, unknown_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            file = f.read()
        X.append(file)
        y.append(unknown_file.split('-')[1])

    return X, y

def calculate_f1_scores(y_true, y_pred, classes, return_precision_recall=False):
    tp = defaultdict(int) # true positives
    fp = defaultdict(int) # false positives
    fn = defaultdict(int) # false negatives

    for true_labels, pred_label in zip(y_true, y_pred):

        for label in classes:
            if label == pred_label:
                if label in true_labels:
                    tp[label] += 1
                else:
                    fp[label] += 1
            elif label in true_labels:
                fn[label] += 1

    f1_scores = []
    precisions = []
    recalls = []
    total_tp, total_fp, total_fn = 0, 0, 0
    for label in classes:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] > 0 else 0

        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1_score)
        precisions.append(precision)
        recalls.append(recall)

        total_tp += tp[label]
        total_fp += fp[label]
        total_fn += fn[label]

    macro_f1 = sum(f1_scores) / len(f1_scores)
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)

    micro_precision = total_tp / (total_tp + total_fp)
    micro_recall = total_tp / (total_tp + total_fn)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    # print all scores in one line formatted nicely to new line
    print("Macro F1: {:.3f}\nMacro Precision: {:.3f}\nMacro Recall: {:.3f}\nMicro F1: {:.3f}\nMicro Precision: {:.3f}\nMicro Recall: {:.3f}".format(macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall))

    if(return_precision_recall): return micro_f1, macro_f1, micro_precision, micro_recall, macro_precision, macro_recall
    else: return micro_f1, macro_f1

def output_y_pred(y_pred, problem_folder):
    predictions = []
    for i, y in enumerate(y_pred):
        entry = {
            "unknown-text": "unknown{}.txt".format(str(i+1).zfill(5)),
            "predicted-author": y
        }
        predictions.append(entry)

    with open(os.path.join(problem_folder, 'answers-{}.json'.format(problem_folder.split("/")[1])), 'w') as f:
        json.dump(predictions, f)

def train(problem_folder, vectorizer, classifier):
    X_train, y_train = load_train_data(problem_folder)
    X_train = vectorizer.fit_transform(X_train)

    #X_train = combine_features(X_train, vectorizers, fit=True)

    max_abs_scaler = preprocessing.MaxAbsScaler()
    scaled_train_data = max_abs_scaler.fit_transform(X_train)
    
    classifier=CalibratedClassifierCV(OneVsRestClassifier(classifier), method='sigmoid', n_jobs=-1)

    classifier.fit(scaled_train_data, y_train)
    # best_clf = classifier.best_estimator_

    # print("Best params: ", classifier.best_params_)
    # print("Best score: ", classifier.best_score_)
    # print("Best clf: ", best_clf)

    return classifier, max_abs_scaler

def test(problem_folder, vectorizer, clf, max_abs_scaler):
    X_test, y_test = load_test_data(problem_folder)
    X_test = vectorizer.transform(X_test)

    #X_test = combine_features(X_test, vectorizers)

    scaled_test_data = max_abs_scaler.transform(X_test)

    # y_pred = clf.predict(scaled_test_data)
    y_pred_proba = clf.predict_proba(scaled_test_data)
    y_pred = []

    for proba in y_pred_proba:
        max_proba_index = proba.argmax()
        if proba[max_proba_index] >= 0.25:
            y_pred.append(clf.classes_[max_proba_index])
        else:
            y_pred.append('<UNK>')
    
    classes = clf.classes_.tolist()
    classes.append('<UNK>')

    calculate_f1_scores(y_test, y_pred, classes, return_precision_recall=True)
    output_y_pred(y_pred, problem_folder)
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)

def run_experiment(vectorizer, classifier, problem_folder):
    print(f"Vectorizer: {type(vectorizer).__name__}")
    print(f"Classifier: {type(classifier).__name__}")

    clf, max_abs = train(problem_folder, vectorizer, classifier)
    accuracy, report = test(problem_folder, vectorizer, clf, max_abs)

    print("Accuracy: ", accuracy)
    print(report)

# def combine_features(X, vectorizers, fit=False):
#     features = []
#     for vec in vectorizers:
#         if fit:
#             features.append(vec.fit_transform(X))
#         else:
#             features.append(vec.transform(X))
#     return hstack(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PAN2019")
    parser.add_argument("--input_base_path", "-i", help="Base path of the input dataset")

    args = parser.parse_args()

    # vectorizers = [
    #     TfidfVectorizer(ngram_range=(1,2), max_features=10000, max_df=0.5, norm='l2', sublinear_tf=True),
    #     CountVectorizer(ngram_range=(1,2), max_features=10000, max_df=0.5)
    # ]

    # classifiers = [
    #     SVC(kernel='linear'),
    #     SVC(C=10, kernel='sigmoid', gamma='scale'),
    #     RandomForestClassifier(n_estimators=100),
    #     GradientBoostingClassifier(),
    #     LogisticRegression(max_iter=1000)
    # ]

    vectorizers = [
        TfidfVectorizer(ngram_range=(1, 3), max_features=15000, max_df=0.5, min_df=2, norm='l2', sublinear_tf=True),
    ]
    #TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=10000, max_df=0.5, norm='l2', sublinear_tf=True)

    # param_grid = {
    #     'C': [0.1, 1, 10, 100, 1000],
    #     'gamma': ['scale', 'auto', 1, 0.1, 0.001, 0.0001],
    #     'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    #     'class_weight': ['balanced', None],
    #     'decision_function_shape': ['ovo', 'ovr'],
    #     'shrinking': [True, False],
    #     'probability': [True, False],
    #     'degree': [2, 3, 4],
    #     'coef0': [0.0, 0.1, 0.5, 1.0]
    # }

    # param_grid = {'C': [10], 'class_weight': ['balanced'], 'coef0': [0.0], 'decision_function_shape': ['ovo'], 'degree': [2], 'gamma': [1], 'kernel': ['sigmoid'], 'probability': [True], 'shrinking': [True]}

    #clf = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1, scoring='f1_macro', refit=True)
    clf = SVC(C=1, class_weight='balanced', coef0=0.0, decision_function_shape='ovo', degree=2, gamma="auto", kernel='rbf', probability=True, shrinking=True)
    
    for vectorizer in vectorizers:
        run_experiment(vectorizer, clf, args.input_base_path)
        print("\n---\n")


    