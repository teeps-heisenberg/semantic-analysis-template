# Sentiment Analysis Project

A comprehensive sentiment analysis system that uses Natural Language Processing (NLP) and Machine Learning to classify movie reviews as positive or negative.

## 📊 Project Overview

This project demonstrates a complete sentiment analysis pipeline using:
- **NLTK** for text preprocessing and tokenization
- **spaCy** for advanced NLP tasks and lemmatization
- **scikit-learn** for machine learning (TF-IDF vectorization and Logistic Regression)
- **pandas** for data manipulation
- **matplotlib & seaborn** for visualization

## 🎯 Features

- **Text Preprocessing**: HTML tag removal, lowercase conversion, punctuation removal, stop word filtering
- **Advanced NLP**: Tokenization, lemmatization using spaCy
- **Feature Engineering**: TF-IDF vectorization with 5000 most common features
- **Machine Learning**: Logistic Regression classifier for sentiment prediction
- **Model Evaluation**: Accuracy metrics, classification report, and confusion matrix visualization
- **Interactive Prediction**: Function to predict sentiment for new text inputs

## 📁 Project Structure

```
semantic-analysis-template/
├── sample.ipynb          # Main Jupyter notebook with complete analysis
├── data.csv              # Dataset with 50,000 movie reviews
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── venv/                 # Virtual environment (ignored by git)
```

## 📈 Dataset Information

- **Size**: 50,000 movie reviews
- **Columns**: 
  - `review`: Text content of the review
  - `sentiment`: Binary classification (positive/negative)
- **Balance**: Perfectly balanced dataset (25,000 positive, 25,000 negative reviews)
- **Memory Usage**: ~781.4 KB

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd semantic-analysis-template
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install nltk pandas spacy scikit-learn matplotlib seaborn jupyter
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

5. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 📖 Usage

### Running the Notebook

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Open `sample.ipynb`** and run all cells sequentially

### Using the Sentiment Predictor

```python
# Example usage of the predict_sentiment function
review = "This movie was absolutely fantastic! The acting was superb."
sentiment = predict_sentiment(review)
print(f"Sentiment: {sentiment}")
# Output: Sentiment: Positive
```

## 🔧 Key Components

### Text Preprocessing Pipeline
```python
def preprocess_text(text):
    """
    1. Removes HTML tags
    2. Lowercases text
    3. Removes punctuation and numbers
    4. Tokenizes text
    5. Removes stop words
    6. Lemmatizes words using spaCy
    """
```

### Machine Learning Pipeline
- **Feature Extraction**: TF-IDF vectorization (5000 features)
- **Model**: Logistic Regression with liblinear solver
- **Train/Test Split**: 80/20 split with stratification
- **Evaluation**: Accuracy, precision, recall, F1-score

## 📊 Results

### Model Performance
- **Accuracy**: 73.00%
- **Precision**: 72-75% (depending on class)
- **Recall**: 64-81% (depending on class)
- **F1-Score**: 69-76% (depending on class)

### Sample Predictions
```
Review: "This movie was absolutely fantastic! The acting was superb and the plot was gripping."
Predicted Sentiment: Positive

Review: "I was so bored throughout the entire film. It was a complete waste of time and money."
Predicted Sentiment: Negative
```

## 🛠️ Technical Details

### Dependencies
- `nltk>=3.8` - Natural Language Toolkit
- `pandas>=1.5.0` - Data manipulation
- `spacy>=3.0.0` - Advanced NLP
- `scikit-learn>=1.1.0` - Machine learning
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization

### Model Architecture
- **Vectorizer**: TF-IDF with 5000 max features
- **Classifier**: Logistic Regression (liblinear solver)
- **Cross-validation**: Stratified train-test split
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, confusion matrix

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- NLTK team for the comprehensive NLP toolkit
- spaCy team for the advanced NLP library
- scikit-learn team for the machine learning framework
- The movie review dataset contributors

---

⭐ **Star this repository if you found it helpful!** 