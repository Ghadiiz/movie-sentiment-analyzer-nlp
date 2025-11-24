# AI Prompt Log for NLP Project

This document contains all prompts used with the integrated AI in Google Colab.

---

## Phase 1: Setup & Data Loading

### Prompt #1: Import Libraries and Setup Environment
Create a Python function that imports all necessary libraries for NLP sentiment analysis including pandas, numpy, matplotlib, seaborn, nltk, sklearn, and tensorflow/keras. Also include functions to download NLTK data packages (stopwords, punkt, wordnet, omw-1.4). Display confirmation messages for each download. Set up the environment for reproducibility with random seeds.

### Prompt #2: Load IMDB Dataset
Write Python code to load the IMDB movie review dataset from keras.datasets. Convert the integer sequences back to text using the word index, handling the offset correctly. Create a pandas dataframe with columns 'review' (the actual text) and 'sentiment' (1 for positive, 0 for negative). Display the first 5 rows, dataframe shape, data types, and basic statistics about the dataset size.

### Prompt #3: Exploratory Data Analysis
Write Python code to perform comprehensive exploratory data analysis on the imdb_df dataframe. Create: 1) A bar chart showing distribution of positive vs negative reviews with exact counts, 2) A histogram showing distribution of review lengths in words with statistics, 3) Check for missing values and duplicates, 4) Display 3 random sample reviews with their sentiments showing both short and long reviews. Use matplotlib and seaborn for professional visualizations with proper titles and labels.

---

## Phase 2: Text Preprocessing with Unit Tests

### Prompt #4: clean_text() Function with Unit Tests
Write Python code to create a clean_text() function that: 1) Replaces <br> tags with spaces before general HTML removal, 2) Removes all HTML tags using BeautifulSoup, 3) Converts to lowercase, 4) Removes special characters keeping only letters and spaces, 5) Removes extra whitespace. Include a detailed docstring. Create 3 unit tests: Test 1 for HTML tags and special characters, Test 2 for mixed case letters, Test 3 specifically for <br> and <br/> tags with extra whitespace. Display input, expected output, actual output, and PASSED/FAILED status for each test with a summary.

### Prompt #5: remove_stopwords() Function with Unit Tests
Write Python code to create a remove_stopwords() function that removes English stopwords using NLTK. Add 'br' as a custom stopword to handle IMDB dataset artifacts. Include a docstring explaining the custom stopword. Create 3 unit tests: Test 1 with common stopwords, Test 2 with mixed content words and stopwords, Test 3 specifically testing the 'br' artifact removal. Display input, expected output, actual output, and PASSED/FAILED status for each test with a summary.

### Prompt #6: lemmatize_text() Function with Unit Tests
Write Python code to create a lemmatize_text() function that applies WordNet lemmatization using NLTK. Include a docstring. Create 3 unit tests: Test 1 with verbs in different tenses, Test 2 with plural nouns, Test 3 with adjectives and adverbs. Show input, expected output, actual output, and PASSED/FAILED status for each test. Display a summary showing how many tests passed.

### Prompt #7: preprocess_pipeline() with Integration Test
Write Python code to create a preprocess_pipeline() function that combines all three preprocessing steps in order: clean_text, remove_stopwords, then lemmatize_text. Include a comprehensive docstring. Create an integration test using a full movie review example with HTML tags, <br> tags, special characters, stopwords, and various word forms. Show step-by-step transformation after each function (Step 1: after clean_text, Step 2: after remove_stopwords, Step 3: after lemmatize_text). Verify the complete pipeline produces the same final output and display PASSED/FAILED verification.

### Prompt #8: Apply Preprocessing to Dataset
Write Python code to apply the preprocess_pipeline() function to all reviews in the imdb_df dataframe. Create a new column called 'processed_review'. Display progress messages, total reviews processed, and show before/after examples for 3 random reviews. For each example, show the original review (first 100 chars), the processed review (first 100 chars), and confirm the transformation worked correctly.

### Prompt #9: Word Frequency Visualization
Write Python code to visualize the top 20 most common words after preprocessing. Use Counter from collections to get word frequencies from all processed reviews. Create a bar chart with proper titles, labels, and formatting. This visualization should verify that no HTML artifacts (like 'br') appear in the most common words, confirming successful preprocessing.

---

## Phase 3: Data Splitting and Feature Engineering

### Prompt #10: Train/Validation/Test Split with TF-IDF
Write Python code to split the dataset into training (70%), validation (15%), and test (15%) sets with stratification on sentiment. Use train_test_split twice: first to separate test set, then to split remaining data into train and validation. Display the exact number and percentage of samples in each set. Then apply TfidfVectorizer with max_features=5000, fitting only on training data to prevent data leakage. Transform all three sets. Display shapes of X_train_tfidf, X_val_tfidf, X_test_tfidf and corresponding y sets. Show vocabulary size and a sample of the TF-IDF matrix.

---

## Phase 4: Model Training with Validation

### Prompt #11: Train Naive Bayes with Validation Evaluation
Write Python code to train a Multinomial Naive Bayes classifier. Fit on training data and make predictions on all three sets (train, validation, test). Calculate and display accuracy for each set with percentages. Display classification reports for both validation and test sets with target names ['Negative', 'Positive']. Use professional formatting with clear section headers showing this is Model 1.

### Prompt #12: Train Logistic Regression with Validation Evaluation
Write Python code to train a Logistic Regression classifier with max_iter=1000 and random_state=42. Fit on training data and make predictions on train, validation, and test sets. Calculate and display accuracy for all three sets with percentages. Display classification reports for validation and test sets. Use professional formatting with clear headers showing this is Model 2.

### Prompt #13: Train Random Forest with Validation Evaluation
Write Python code to train a Random Forest classifier with n_estimators=100, random_state=42, and n_jobs=-1 for parallel processing. Include a message that training may take a few minutes. Fit on training data and make predictions on all three sets. Calculate and display accuracy for train, validation, and test with percentages. Display classification reports for validation and test sets. Use professional formatting showing this is Model 3.

---

## Phase 5: Deep Learning Model

### Prompt #14: Prepare Data for LSTM with Proper Split
Write Python code to prepare data for LSTM training using the existing train/validation/test split. Initialize a Tokenizer with num_words=5000 and maxlen=200. Fit the tokenizer only on X_train to prevent data leakage. Convert all three sets (X_train, X_val, X_test) to sequences and pad them. Convert labels to numpy arrays. Display shapes of X_train_lstm, X_val_lstm, X_test_lstm. Show vocabulary size and a sample padded sequence. Confirm data preparation is complete with no data leakage.

### Prompt #15: Build and Train LSTM Model
Write Python code to build an LSTM model using Keras Sequential API with: 1) Embedding layer (input_dim=5000, output_dim=128, input_length=200), 2) LSTM layer (128 units, dropout=0.2, recurrent_dropout=0.2), 3) Dense layer with sigmoid activation. Compile with adam optimizer and binary_crossentropy loss. Display model summary. Train for 5 epochs with batch_size=64 using validation_data=(X_val_lstm, y_val_lstm) NOT validation_split. Display training progress. Create two plots side-by-side: training vs validation accuracy, and training vs validation loss over epochs. Evaluate on all three sets and display train, validation, and test accuracy with percentages. Use professional formatting showing this is Model 4.

---

## Phase 6: Comprehensive Model Comparison

### Prompt #16: Complete Model Comparison with Validation and Test
Write Python code to create a comprehensive comparison of all 4 models. First, calculate LSTM predictions and metrics (accuracy, precision, recall, f1-score) for both validation and test sets. Create two comparison tables: Table 1 showing validation performance for all 4 models, Table 2 showing test performance. Identify the best model based on validation accuracy. Create side-by-side bar charts comparing validation and test accuracy for all models with exact values displayed. Show confusion matrix for LSTM on test set using seaborn heatmap. Include a final summary explaining the selected model, its validation accuracy, test accuracy, and methodology used for selection.

---

## Phase 7: Application Development

### Prompt #17: Interactive Sentiment Analyzer
Write Python code to create a sentiment analysis function called predict_sentiment() that takes raw text input, applies the preprocess_pipeline(), vectorizes using the fitted tfidf_vectorizer, and predicts using the best model (Logistic Regression). Return sentiment label and confidence percentage. Include error handling for empty input. Create 5 diverse example reviews (very positive, very negative, mixed, different styles) and test the function on each. Display results in a formatted way showing the review text, predicted sentiment, and confidence percentage. Provide instructions for using the function with custom text.

### Prompt #18: Project Conclusion and Analysis
Write Python code to create a comprehensive project conclusion including: 1) Project overview listing all 4 models, 2) Methodology section explaining the 70/15/15 split, preprocessing pipeline with unit tests, and feature engineering approaches, 3) Performance summary table showing validation and test accuracy for all models, 4) Model selection explanation based on validation accuracy with justification, 5) Detailed explanation of how the selected model works from input to output, 6) Five specific limitations (domain specificity, binary classification, context understanding, language limitation, aspect-level analysis), 7) Seven future improvements (advanced embeddings, multi-class sentiment, aspect-based analysis, cross-domain transfer, hyperparameter optimization, explainable AI, production deployment). Format with clear headers, proper spacing, and professional presentation. End with a project complete message and summary statement.

---

## Summary

**Total Prompts Used:** 18

**Project Phases:**
- Phase 1: Setup & Data Loading (Prompts 1-3)
- Phase 2: Preprocessing with Unit Tests (Prompts 4-9)
- Phase 3: Data Splitting & Feature Engineering (Prompt 10)
- Phase 4: Classical ML Models with Validation (Prompts 11-13)
- Phase 5: Deep Learning Model (Prompts 14-15)
- Phase 6: Comprehensive Evaluation (Prompt 16)
- Phase 7: Application Development (Prompts 17-18)

**Models Implemented:** 4 (Naive Bayes, Logistic Regression, Random Forest, LSTM)

**Dataset Split:** 70% Train / 15% Validation / 15% Test

**Best Model:** Logistic Regression (selected based on validation accuracy)

**Key Features:**
- Comprehensive unit tests for all preprocessing functions
- Integration test for complete pipeline
- Proper train/validation/test split to prevent data leakage
- Validation-based model selection following ML best practices
- 'br' artifact handling specific to IMDB dataset
- Professional formatting and documentation throughout

