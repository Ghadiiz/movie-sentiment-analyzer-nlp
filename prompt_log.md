# AI Prompt Log for NLP Project

This document contains all prompts used with the integrated AI in Google Colab.

---

## Phase 1: Setup & Data Loading

### Prompt #1: Import Libraries
1.	Create a Python function that imports all necessary libraries for NLP sentiment analysis including pandas, numpy, matplotlib, seaborn, nltk, sklearn, and tensorflow/keras. Also include functions to download NLTK data packages (stopwords, punkt, wordnet).

### Prompt #2: Load Dataset
2.	Write Python code to load the IMDB movie review dataset from keras.datasets. Convert the integer sequences back to text using the word index. Create a pandas dataframe with columns 'review' (the actual text) and 'sentiment' (1 for positive, 0 for negative). Display the first 5 rows, dataframe shape, and data types. Include all necessary code without instructions or comments explaining alternatives.

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
[Prompts will be added as project progresses]
