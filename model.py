from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import argparse
import os

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
        y.append(unknown_file.split('-')[0])

    return X, y


def train(problem_folder, vectorizer, svm):
    X_train, y_train = load_train_data(problem_folder)
    X_train = vectorizer.fit_transform(X_train)

    svm.fit(X_train, y_train)

    return svm

def test(problem_folder, vectorizer, svm):
    X_test, y_test = load_test_data(problem_folder)
    X_test = vectorizer.transform(X_test)

    y_pred = svm.predict(X_test)
    
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PAN2019")
    parser.add_argument("--input_base_path", "-i", help="Base path of the input dataset")

    args = parser.parse_args()

    vectorizer = TfidfVectorizer()
    svm = SVC(kernel='linear')

    svm = train(args.input_base_path, vectorizer, svm)
    accuracy, report = test(args.input_base_path, vectorizer, svm)

    print("Accuracy: ", accuracy)
    print(report)

    