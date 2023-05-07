import os
import re
import string
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk import ngrams
from collections import Counter
import json

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

def load_unknown_data(problem_folder):
    unknown_folder = os.path.join(problem_folder, 'unknown')
    ground_truth_file = os.path.join(problem_folder, 'ground-truth.json')

    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)['ground_truth']

    label_dict = {entry['unknown-text']: entry['true-author'] for entry in ground_truth}

    data = []

    for filename in os.listdir(unknown_folder):
        file_path = os.path.join(unknown_folder, filename)
        author = label_dict[filename]
        data.append((file_path, author))

    return data


def preprocess(text, min_word_length=3, min_word_freq=3, ngram_range=(1, 3)):
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

    # replace all integers and floats with <NUM>
    words = [re.sub(r'\d+.\d+|\d+', '<NUM>', w) for w in words]

    # Apply Porter stemming
    #stemmer = PorterStemmer()
    #words = [stemmer.stem(w) for w in words]

    # Remove short words
    #words = [w for w in words if len(w) >= min_word_length]

    # Apply Snowball stemming
    # stemmer = SnowballStemmer("english")
    # words = [stemmer.stem(w) for w in words]

    # Generate n-grams
    # ngrams_list = []
    # for n in range(ngram_range[0], ngram_range[1] + 1):
    #     ngrams_list.extend(ngrams(words, n))

    # ngrams_joined = [' '.join(ngram) for ngram in ngrams_list]

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
        problem_name = os.path.basename(problem_folder)
        output_problem_path = os.path.join(output_base_path, problem_name)
        
        # Process candidate files
        data = load_data(problem_folder)

        if not os.path.exists(output_problem_path):
            os.makedirs(output_problem_path)

        for input_file, author_folder in data:
            output_author_path = os.path.join(output_problem_path, author_folder)

            if not os.path.exists(output_author_path):
                os.makedirs(output_author_path)

            output_file = os.path.join(output_author_path, os.path.basename(input_file))
            process_and_save_file(input_file, output_file)

        # Process unknown files
        unknown_data = load_unknown_data(problem_folder)
        output_unknown_path = os.path.join(output_problem_path, 'unknown')

        if not os.path.exists(output_unknown_path):
            os.makedirs(output_unknown_path)

        for input_file, author in unknown_data:
            output_file = os.path.join(output_unknown_path, os.path.basename(input_file) + '-' + author)
            process_and_save_file(input_file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PAN2019")
    parser.add_argument("--input_base_path", "-i", help="Base path of the input dataset")
    parser.add_argument("--output_base_path", "-o", help="Base path for the preprocessed dataset")

    args = parser.parse_args()

    preprocess_and_save_files(args.input_base_path, args.output_base_path)