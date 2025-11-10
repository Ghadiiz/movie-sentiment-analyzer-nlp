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
[Prompts will be added as project progresses]

---

## Phase 4: Model Evaluation
[Prompts will be added as project progresses]

---

## Phase 5: Application Development
[Prompts will be added as project progresses]
