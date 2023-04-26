import os
import re
import string
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the NLTK stop words if you haven't already
nltk.download('stopwords')


def load_data(problem_folder):
    author_folders = [f for f in os.listdir(problem_folder) if f.startswith("candidate")]
    data = []

    for author_folder in author_folders:
        author_path = os.path.join(problem_folder, author_folder)
        author_files = os.listdir(author_path)

        for author_file in author_files:
            file_path = os.path.join(author_path, author_file)
            data.append((file_path, author_folder))

    return data


def preprocess(text):
    # Handle punctuation
    translator = str.maketrans('/-', '  ', string.punctuation.translate(str.maketrans('', '', '/-.')))
    text = text.translate(translator)
    text = re.sub(r"\.(?!\d)", "", text)

    # Tokenize the text into words and handle case
    words = re.findall(r'\b\w+\b', text.casefold())

    # Remove extra spaces
    words = [w.strip() for w in words]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    # Apply Porter stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)


def process_and_save_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    preprocessed_text = preprocess(text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(preprocessed_text)


def preprocess_and_save_files(input_base_path, output_base_path):
    problem_folders = [os.path.join(input_base_path, d) for d in os.listdir(input_base_path) if (d.startswith("problem") and int(d[-2:]) < 6)]

    for problem_folder in problem_folders:
        data = load_data(problem_folder)
        problem_name = os.path.basename(problem_folder)
        output_problem_path = os.path.join(output_base_path, problem_name)

        if not os.path.exists(output_problem_path):
            os.makedirs(output_problem_path)

        for input_file, author_folder in data:
            output_author_path = os.path.join(output_problem_path, author_folder)

            if not os.path.exists(output_author_path):
                os.makedirs(output_author_path)

            output_file = os.path.join(output_author_path, os.path.basename(input_file))
            process_and_save_file(input_file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PAN2019")
    parser.add_argument("--input_base_path", "-i", help="Base path of the input dataset")
    parser.add_argument("--output_base_path", "-o", help="Base path for the preprocessed dataset")

    args = parser.parse_args()

    preprocess_and_save_files(args.input_base_path, args.output_base_path)