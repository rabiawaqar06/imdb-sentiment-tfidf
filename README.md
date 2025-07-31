# IMDB Movie Reviews Sentiment Analysis

## Overview
This project implements a machine learning model to analyze sentiment in movie reviews using the IMDB Dataset of 50K Movie Reviews. The model uses Natural Language Processing (NLP) techniques to classify movie reviews as either positive or negative, achieving 80% accuracy on the test set.

## Dataset
The project uses the IMDB Dataset containing 50,000 movie reviews, balanced between positive and negative sentiments. Due to size limitations, the dataset is not included in this repository but can be downloaded from:
[IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Features
- Text preprocessing with NLTK and spaCy
- TF-IDF vectorization for feature extraction
- Logistic Regression model for classification
- Detailed performance metrics and visualizations
- Real-time sentiment prediction for new reviews

## Technical Details

### Preprocessing Pipeline
- HTML tag removal
- Text lowercasing
- Punctuation and number removal
- Tokenization
- Stop word removal
- Lemmatization using spaCy

### Model Architecture
- TF-IDF Vectorization (max_features=5000)
- Logistic Regression with liblinear solver
- 80-20 train-test split
- Stratified sampling for balanced classes

### Performance Metrics
- Accuracy: 80%
- Balanced performance for both positive and negative classes
- Detailed confusion matrix and classification report

## Requirements
```
pandas
nltk
spacy
scikit-learn
seaborn
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. Download spaCy model:
```python
python -m spacy download en_core_web_sm
```

5. Download the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project root directory.

## Usage

1. Open and run the Jupyter notebook `sentiment-analysis.ipynb`
2. The notebook contains step-by-step implementation including:
   - Data loading and preprocessing
   - Model training
   - Evaluation
   - Example predictions

Example usage of the sentiment predictor:
```python
review = "This movie was absolutely fantastic! The acting was superb."
sentiment = predict_sentiment(review)
print(f"Predicted sentiment: {sentiment}")
```

## Results
- 80% accuracy on the test set
- Balanced precision and recall for both classes
- Robust performance on various types of reviews

## Future Improvements
- Implement more advanced NLP techniques
- Try different model architectures (BERT, RoBERTa)
- Add cross-validation
- Expand to multi-class sentiment analysis
- Add confidence scores for predictions

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- IMDB Dataset providers
- NLTK and spaCy libraries
- Scikit-learn community
