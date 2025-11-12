# AI Prompt Log for NLP Project

This document contains all prompts used with the integrated AI in Google Colab.

---

## Phase 1: Setup & Data Loading

### Prompt #1: Import Libraries
Create a Python function that imports all necessary libraries for NLP sentiment analysis including pandas, numpy, matplotlib, seaborn, nltk, sklearn, and tensorflow/keras. Also include functions to download NLTK data packages (stopwords, punkt, wordnet).

### Prompt #2: Load Dataset
Write Python code to load the IMDB movie review dataset from keras.datasets. Convert the integer sequences back to text using the word index. Create a pandas dataframe with columns 'review' (the actual text) and 'sentiment' (1 for positive, 0 for negative). Display the first 5 rows, dataframe shape, and data types. Include all necessary code without instructions or comments explaining alternatives.

### Prompt #3: Data Exploration
Write Python code to perform exploratory data analysis on a dataframe called 'imdb_df' with columns 'review' and 'sentiment'. Create: 1) A bar chart showing count of positive (sentiment=1) vs negative (sentiment=0) reviews, 2) A histogram showing distribution of review lengths in words, 3) Check for missing values, 4) Display 5 random sample reviews with their sentiments. Use matplotlib and seaborn for visualizations.

---

## Phase 2: Text Preprocessing

### Prompt #4: Text Cleaning
Write a Python function called clean_text that takes a text string and performs these preprocessing steps: 1) Convert to lowercase, 2) Remove HTML tags using BeautifulSoup or regex, 3) Remove special characters and punctuation keeping only letters and spaces, 4) Remove extra whitespaces. Apply this function to all reviews in the imdb_df dataframe and create a new column called 'cleaned_review'. Display before and after examples of 3 reviews.

### Prompt #5: Tokenization and Stopword Removal
Write Python code to tokenize the cleaned reviews using NLTK. Remove stopwords using NLTK's English stopwords list. Apply lemmatization using WordNetLemmatizer. Create a new column in imdb_df called 'processed_review' containing the processed text. Show the difference between cleaned_review and processed_review for 3 examples. Also display the 20 most common words after processing using a bar chart.

### Prompt #6: Train-Test Split and Vectorization
Write Python code to split the imdb_df dataset into training (80%) and testing (20%) sets with stratification on sentiment. Use TfidfVectorizer from sklearn with max_features=5000 to convert text to numerical features. Fit the vectorizer on training data only and transform both train and test sets. Display the shapes of X_train, X_test, y_train, y_test and the vocabulary size. Also show a sample of the TF-IDF matrix.

---

## Phase 3: Model Training

### Prompt #7: Train Naive Bayes Model
Write Python code to train a Multinomial Naive Bayes classifier on the training data (X_train_tfidf and y_train). Import the model from sklearn.naive_bayes. Fit the model and make predictions on both training and test sets. Calculate and display accuracy scores for both train and test. Also display a classification report for the test set showing precision, recall, and f1-score.

### Prompt #8: Train Logistic Regression Model
Write Python code to train a Logistic Regression classifier on the training data. Use sklearn.linear_model.LogisticRegression with max_iter=1000. Fit the model and make predictions on test data. Display training accuracy, test accuracy, and a detailed classification report with precision, recall, f1-score for both classes.

### Prompt #9: Train Random Forest Model
Write Python code to train a Random Forest classifier on the training data using sklearn.ensemble.RandomForestClassifier with n_estimators=100 and random_state=42. Train the model and make predictions. Display training accuracy, test accuracy, and classification report. Note: This may take a few minutes to train.

---

## Phase 4: Model Evaluation

### Prompt #10: Confusion Matrices for All Models
Write Python code to create confusion matrices for all three models (Naive Bayes, Logistic Regression, Random Forest). Display them side by side in a single figure with 3 subplots using seaborn heatmaps. Each heatmap should show the confusion matrix with annotations and a title indicating which model it represents. Use the test set predictions for all models.

### Prompt #11: Model Comparison Chart
Write Python code to create a comparison visualization showing accuracy, precision, recall, and f1-score for all three models (Naive Bayes, Logistic Regression, Random Forest). Create a grouped bar chart with models on x-axis and metrics on y-axis. Use different colors for each metric and include a legend. Display both training and test accuracy in a separate comparison chart.

### Prompt #12: ROC Curves
Write Python code to plot ROC curves for all three models on the same graph. Calculate and display the AUC (Area Under Curve) score for each model. Include a diagonal reference line representing random guessing. Add a legend showing each model's AUC score. Use the test set for evaluation.

---

## Phase 5: Application Development

### Prompt #13: Interactive Sentiment Predictor
Write Python code to create an interactive sentiment analysis application. Create a function that takes a user's text input, applies the same preprocessing steps (cleaning, tokenization, stopword removal, lemmatization), vectorizes it using the trained TF-IDF vectorizer, and predicts sentiment using the Logistic Regression model. Display the prediction (Positive/Negative) with a confidence score as a percentage. Create a simple text input interface using ipywidgets or allow the user to test with sample reviews. Include at least 3 example reviews to test.

### Prompt #14: Model Summary and Conclusion
Write Python code to create a comprehensive project summary including: 1) A markdown table comparing all three models with their accuracy, precision, recall, f1-score, and AUC scores, 2) A text conclusion explaining which model performed best and why (Logistic Regression), 3) Limitations of the current approach (e.g., dataset size, binary classification only, no aspect-based sentiment), 4) Future improvements (e.g., using BERT/transformers, multi-class sentiment, real-time data collection). Format this as a nicely displayed output with proper headings.

---

## Phase 6: Deep Learning Model (Professor Feedback - Requirement #1)

### Prompt #15: Prepare Data for LSTM
Write Python code to prepare the IMDB dataset for LSTM training. Use the Tokenizer from tensorflow.keras.preprocessing.text to tokenize the processed reviews. Convert texts to sequences, then pad sequences to a maximum length of 200. Create X_train_lstm, X_test_lstm variables from the processed_review column. Display the shape of the padded sequences and show a sample sequence. Also save the tokenizer for later use in predictions.

### Prompt #16: Build and Train LSTM Model
Write Python code to build and train an LSTM model for sentiment classification using Keras. Create a Sequential model with: 1) Embedding layer with 5000 vocabulary size and 128 dimensions, 2) LSTM layer with 128 units and dropout 0.2, 3) Dense layer with sigmoid activation for binary classification. Compile with adam optimizer and binary_crossentropy loss. Train for 5 epochs with batch_size=64 and validation_split=0.2. Display training history with accuracy and loss plots. Then evaluate on test set and display test accuracy.

### Prompt #17: Evaluate LSTM with Other Models
Write Python code to evaluate the LSTM model and compare it with the previous three models (Naive Bayes, Logistic Regression, Random Forest). Generate predictions, calculate accuracy, precision, recall, f1-score. Create a confusion matrix for LSTM. Then create an updated comparison table showing all 4 models with their performance metrics. Also update the ROC curve plot to include LSTM as the 4th curve.

---

## Phase 7: Code Refactoring with Unit Tests (Professor Feedback - Requirement #2)

### Prompt #18: clean_text() Function + Unit Tests
Write Python code to create a clean_text() function that removes HTML tags using BeautifulSoup, converts text to lowercase, removes special characters keeping only letters and spaces, and removes extra whitespace. Include a docstring. Then create 3 unit tests: Test 1 with HTML tags and special characters, Test 2 with mixed case letters, Test 3 with extra whitespace. Show input, expected output, actual output, and PASSED/FAILED status for each test. Display a summary at the end.

### Prompt #19: remove_stopwords() Function + Unit Tests
Write Python code to create a remove_stopwords() function that removes English stopwords from text using NLTK. Include a docstring. Then create 3 unit tests: Test 1 with common stopwords like 'the', 'is', 'a', Test 2 with only content words (no stopwords), Test 3 with an empty string. Show input, expected output, actual output, and PASSED/FAILED status for each test. Display a summary.

### Prompt #20: lemmatize_text() Function + Unit Tests
Write Python code to create a lemmatize_text() function that applies WordNet lemmatization to convert words to their base form. Include a docstring. Then create 3 unit tests with different word forms: Test 1 with verbs and adjectives, Test 2 with plural nouns, Test 3 with various word forms. Show input, actual output (lemmatization varies), and execution status. Display a summary.

### Prompt #21: preprocess_pipeline() + Integration Test
Write Python code to create a preprocess_pipeline() function that combines all preprocessing steps: clean_text, remove_stopwords, and lemmatize_text. Include a docstring. Then create an integration test with a full movie review example containing HTML tags, special characters, stopwords, and various word forms. Show the original text, step-by-step transformation after each function, and final output. Verify that the pipeline function produces the same result.

---

## Summary

**Total Prompts Used:** 21

**Project Phases:**
- Phase 1-5: Original project implementation (Prompts 1-14)
- Phase 6: LSTM deep learning model addition (Prompts 15-17)
- Phase 7: Code refactoring with comprehensive unit testing (Prompts 18-21)

**Models Implemented:** 4 (Naive Bayes, Logistic Regression, Random Forest, LSTM)

**Best Model:** Logistic Regression (88.69% accuracy, 0.9551 AUC)
