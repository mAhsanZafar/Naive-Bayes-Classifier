# Naive Bayes Gender Classification

A simple Naive Bayes classifier that predicts gender from the last two letters of a name. This project trains a Multinomial Naive Bayes model using character bigrams and provides a command-line interface for predictions.

## Project Overview
This project demonstrates a lightweight approach to gender classification based on name suffixes. It uses scikit-learn’s `CountVectorizer` with character bigrams and a `MultinomialNB` classifier. The implementation is kept intentionally minimal for educational purposes.

## Dataset
The model expects a CSV file named `genders.csv` in the project root with at least the following columns:

- `name` — the person’s name
- `gender` — the target label (e.g., `male`, `female`)

Example:

name,gender
Alex,male
Maria,female

## Method
- Feature: last two letters of each name
- Vectorization: character bigrams (`ngram_range=(2,2)`)
- Classifier: Multinomial Naive Bayes

## Installation
1. Clone the repository
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

## Usage
Ensure `genders.csv` is in the same directory as the script, then run:

```bash
python Gender_clasifiction_Naive_Bayes_Classifier.py
```

You will be prompted to enter a name. Type `exit` to quit.

## Notes
- This is a simple baseline model and may not perform well on diverse or international names.
- For better results, consider richer features (full name, language-specific suffixes) and a larger dataset.

## License
No license specified. Add a LICENSE file if you plan to share or reuse this project publicly.

## Author
Created by mAhsanZafar.
