# Movie Sentiment Analyzer - NLP Project

## Overview
This project implements a comprehensive movie review sentiment analysis system using both traditional machine learning and deep learning approaches. It classifies IMDB movie reviews as positive or negative, demonstrating the full data science pipeline from data preprocessing with unit tests to model evaluation with proper train/validation/test methodology.

## Features
- **Dataset:** 50,000 IMDB movie reviews (balanced: 25,000 positive, 25,000 negative)
- **Data Split:** 70% Training (35,000) / 15% Validation (7,480) / 15% Test (7,500) with stratification
- **Preprocessing:** Complete NLP pipeline with modular functions, comprehensive unit tests, and integration testing
- **Models:** 4 different classifiers implemented and compared
  - Multinomial Naive Bayes
  - Logistic Regression ⭐ **Best Model** (selected via validation set)
  - Random Forest (shows overfitting with 100% training accuracy)
  - LSTM Deep Learning
- **Evaluation:** Comprehensive benchmarking with validation/test comparison, confusion matrices, and performance metrics
- **Testing:** Unit tests for clean_text(), remove_stopwords(), lemmatize_text(), and integration test for preprocess_pipeline()
- **Application:** Interactive sentiment prediction function with confidence scores

## Technologies Used
- **Python 3.x**
- **Libraries:**
  - Data Processing: pandas, numpy
  - NLP: nltk (stopwords, punkt, wordnet, lemmatization)
  - Machine Learning: scikit-learn (Naive Bayes, Logistic Regression, Random Forest, TF-IDF)
  - Deep Learning: TensorFlow/Keras (LSTM, Embedding layers)
  - Visualization: matplotlib, seaborn
  - HTML Parsing: BeautifulSoup (for cleaning review text)

## Methodology

### Data Preprocessing Pipeline
Each preprocessing step includes comprehensive unit tests:

1. **clean_text()** - HTML removal, lowercase conversion, special character removal
   - Handles `<br>` tags specifically (common in IMDB dataset)
   - Unit tests: HTML tags, mixed case, whitespace handling
   - All tests: ✅ PASSED

2. **remove_stopwords()** - NLTK stopword removal
   - Custom stopword: 'br' (artifact from IMDB dataset)
   - Unit tests: common stopwords, mixed content, 'br' removal
   - All tests: ✅ PASSED

3. **lemmatize_text()** - WordNet lemmatization
   - Converts words to base forms
   - Unit tests: verbs, plural nouns, adjectives
   - All tests: ✅ PASSED

4. **preprocess_pipeline()** - Combined pipeline
   - Integration test with full movie review example
   - Verifies all steps work together correctly
   - Test: ✅ PASSED

### Feature Engineering
- **Classical ML Models:** TF-IDF vectorization (max 5,000 features)
  - Fitted only on training data to prevent data leakage
  - Vocabulary size: 5,000 features
- **LSTM Model:** Tokenization and padding (max length 200)
  - Vocabulary size: 5,000 most frequent words
  - Embedding dimension: 128

### Model Training Approach
- **Validation-based selection:** Models evaluated on validation set to select the best performer
- **Test set:** Held out for final unbiased evaluation
- **No data leakage:** All feature engineering fitted only on training data
- **Stratification:** Maintained class balance across all splits

## Results

### Model Performance Comparison

#### Validation Set Performance
| Model | Val Accuracy | Val Precision | Val Recall | Val F1-Score |
|-------|--------------|---------------|------------|--------------|
| Naive Bayes | 85.41% | 0.85 | 0.85 | 0.85 |
| **Logistic Regression** | **88.74%** | **0.89** | **0.89** | **0.89** |
| Random Forest | 85.05% | 0.85 | 0.85 | 0.85 |
| LSTM | 87.58% | 0.88 | 0.88 | 0.88 |

#### Test Set Performance
| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|-------|---------------|----------------|-------------|---------------|
| Naive Bayes | 85.23% | 0.85 | 0.85 | 0.85 |
| **Logistic Regression** | **88.59%** | **0.89** | **0.89** | **0.89** |
| Random Forest | 84.29% | 0.84 | 0.84 | 0.84 |
| LSTM | 87.09% | 0.87 | 0.87 | 0.87 |

### Training Performance (Overfitting Check)
| Model | Training Accuracy | Validation Accuracy | Overfitting? |
|-------|-------------------|---------------------|--------------|
| Naive Bayes | 86.66% | 85.41% | ❌ No (1.25% gap) |
| Logistic Regression | 91.44% | 88.74% | ⚠️ Minimal (2.70% gap) |
| **Random Forest** | **100.00%** | **85.05%** | ✅ **Yes (14.95% gap)** |
| LSTM | 95.25% | 87.58% | ⚠️ Moderate (7.67% gap) |

### Key Findings
1. **Best Model: Logistic Regression**
   - Selected based on highest validation accuracy (88.74%)
   - Strong test performance (88.59%) confirms good generalization
   - Minimal overfitting with only 2.70% training-validation gap
   - Computationally efficient and interpretable

2. **Random Forest Overfitting**
   - Perfect 100% training accuracy but only 85.05% validation accuracy
   - 14.95% gap indicates severe overfitting
   - Despite complexity, underperformed compared to Logistic Regression

3. **LSTM Performance**
   - Achieved 87.58% validation and 87.09% test accuracy
   - Competitive but didn't outperform simpler Logistic Regression
   - 7.67% training-validation gap shows moderate overfitting
   - Requires significantly more computational resources

4. **TF-IDF Effectiveness**
   - Traditional TF-IDF features proved highly effective
   - Simpler models (Naive Bayes, Logistic Regression) performed competitively
   - Proper handling of 'br' artifacts was critical for success

5. **Preprocessing Impact**
   - Unit-tested preprocessing pipeline ensured reliability
   - Removing 'br' artifact significantly improved word frequency distributions
   - Lemmatization and stopword removal enhanced model performance

### Example Predictions
The notebook includes 5 diverse example reviews demonstrating:
- Very positive reviews (high confidence)
- Very negative reviews (high confidence)
- Mixed sentiment reviews (lower confidence)
- Different writing styles and lengths


---


## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- pip (Python package manager)

### Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Ghadiiz/movie-sentiment-analyzer-nlp.git

# Navigate to the project directory
cd movie-sentiment-analyzer-nlp
```

### Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

### Download NLTK Data

The app will automatically download required NLTK data on first run. If you prefer to download manually:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

### Get the Trained Models

The trained models are required to run the Streamlit app. Place these files in the project root directory:
- `logistic_regression_model.pkl`
- `tfidf_vectorizer.pkl`

**Train Your Own Models**
1. Open the Jupyter notebook: `Movie_Sentiment_Analysis.ipynb`
2. Run all cells to train the models
3. Run the export cell at the end to generate the `.pkl` files

## Running the Application

### Start the Streamlit App

```bash
# Make sure you're in the project directory
cd movie-sentiment-analyzer-nlp

# Run the app
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Stopping the Application

Press `Ctrl + C` in the terminal to stop the server.

## Quick Start Guide

```bash
# Complete setup in 4 commands
git clone https://github.com/Ghadiiz/movie-sentiment-analyzer-nlp.git
cd movie-sentiment-analyzer-nlp
pip install -r requirements.txt
streamlit run app.py
```

**Note:** Ensure model files (`.pkl`) are in the project directory before running.

## Troubleshooting

**Issue: "Model files not found" error**
```
⚠️ Solution: Place logistic_regression_model.pkl and tfidf_vectorizer.pkl 
   in the project root directory
```

**Issue: Port already in use**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**Issue: ModuleNotFoundError**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## Documentation

- **Prompt Log:** See `prompt.md` for all 18 AI prompts used in development
- **Code Comments:** Comprehensive inline documentation throughout the notebook
- **Test Results:** All unit tests display PASSED/FAILED status with examples
- **Docstrings:** Every function includes detailed docstrings explaining parameters and returns

## Performance Summary

🏆 **Winner: Logistic Regression**
- **Validation Accuracy:** 88.74%
- **Test Accuracy:** 88.59%
- **Selected for:** Best validation performance, minimal overfitting, computational efficiency

📊 **All Models:**
1. Logistic Regression: 88.59% test accuracy
2. LSTM: 87.09% test accuracy
3. Naive Bayes: 85.23% test accuracy
4. Random Forest: 84.29% test accuracy (despite 100% training accuracy)

---

**Note:** This project demonstrates ML best practices including proper data splitting, validation-based model selection, preventing data leakage, comprehensive testing, and transparent evaluation methodology. The surprising result that classical ML (Logistic Regression) outperformed deep learning (LSTM) highlights the importance of proper model selection and the fact that more complex models don't always guarantee better performance.
