# Movie Sentiment Analyzer - NLP Project

## Overview
This project implements a comprehensive movie review sentiment analysis system using both traditional machine learning and deep learning approaches. It classifies IMDB movie reviews as positive or negative, demonstrating the full data science pipeline from data preprocessing to model deployment.

## Features
- **Dataset:** 50,000 IMDB movie reviews (balanced)
- **Preprocessing:** Complete NLP pipeline with modular functions and unit tests
- **Models:** 4 different classifiers implemented and compared
  - Multinomial Naive Bayes (85.47% accuracy)
  - Logistic Regression (88.69% accuracy) ⭐ **Best Model**
  - Random Forest (84.51% accuracy)
  - LSTM Deep Learning (Deep learning approach)
- **Evaluation:** Comprehensive benchmarking with confusion matrices, ROC curves, and performance metrics
- **Testing:** Unit tests for each preprocessing function
- **Application:** Interactive sentiment prediction with confidence scores


## Technologies Used
- **Python 3.x**
- **Libraries:**
  - Data Processing: pandas, numpy
  - NLP: nltk, scikit-learn (TF-IDF)
  - Machine Learning: scikit-learn (Naive Bayes, Logistic Regression, Random Forest)
  - Deep Learning: TensorFlow/Keras (LSTM)
  - Visualization: matplotlib, seaborn
  - Web Scraping: BeautifulSoup (HTML cleaning)
  - Interactive UI: ipywidgets

## Results

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Naive Bayes | 85.47% | 85.47% | 85.47% | 0.8547 | 0.9313 |
| **Logistic Regression** | **88.69%** | **88.71%** | **88.69%** | **0.8869** | **0.9551** |
| Random Forest | 84.51% | 84.51% | 84.51% | 0.8451 | 0.9270 |
| LSTM | Variable | - | - | - | - |

### Key Findings
- **Logistic Regression** achieved the best overall performance
- Traditional ML models outperformed LSTM (likely due to dataset characteristics)
- TF-IDF vectorization proved highly effective for this task
- Model shows strong generalization with minimal overfitting



### Option 2: Local Environment

