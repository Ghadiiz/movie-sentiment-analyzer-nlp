import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import nltk
import os

# Download NLTK data (first time only)
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.add('br')  # Add IMDB artifact

# ============================================================================
# PREPROCESSING FUNCTIONS (Same as notebook)
# ============================================================================

def clean_text(text):
    """
    Clean text by removing HTML tags, converting to lowercase,
    removing special characters, and removing extra whitespace.
    """
    # Replace <br> tags with spaces BEFORE using BeautifulSoup
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    
    # Remove all other HTML tags using BeautifulSoup
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def remove_stopwords(text):
    """
    Remove stopwords from text.
    """
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """
    Lemmatize words to their base form.
    """
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_pipeline(text):
    """
    Complete preprocessing pipeline combining all steps.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models and vectorizer"""
    try:
        # Load Logistic Regression model
        with open('logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer, True
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please export models from Colab first.")
        return None, None, False

# Load models
lr_model, tfidf_vec, models_loaded = load_models()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sentiment(text):
    """
    Predict sentiment using the trained Logistic Regression model.
    """
    if not models_loaded:
        return "Error: Models not loaded", 0.0
    
    # Preprocess the text
    processed_text = preprocess_pipeline(text)
    
    # Vectorize using TF-IDF
    text_tfidf = tfidf_vec.transform([processed_text])
    
    # Predict using Logistic Regression
    prediction = lr_model.predict(text_tfidf)[0]
    prediction_proba = lr_model.predict_proba(text_tfidf)[0]
    
    # Get sentiment label and confidence
    sentiment_label = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = prediction_proba[prediction] * 100
    
    return sentiment_label, confidence, processed_text

# ============================================================================
# STREAMLIT APP UI
# ============================================================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Header
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("### AI-Powered NLP Sentiment Analysis")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Project Information")
    st.info(
        """
        This application uses **Natural Language Processing** 
        and **Machine Learning** to classify movie reviews.
        
        **Dataset:** 50,000 IMDB Reviews
        
        **Data Split:**
        - Training: 70% (35,000)
        - Validation: 15% (7,480)
        - Test: 15% (7,500)
        
        **Models Trained:**
        1. Naive Bayes - 85.23%
        2. **Logistic Regression - 88.59%** ⭐
        3. Random Forest - 84.29%
        4. LSTM - 87.09%
        
        **Selected Model:** Logistic Regression
        - Validation Acc: 88.74%
        - Test Acc: 88.59%
        - Best balance of accuracy and efficiency
        """
    )
    
    st.header("🔧 Preprocessing Steps")
    st.markdown("""
    1. **HTML Tag Removal**
    2. **Lowercase Conversion**
    3. **Special Character Removal**
    4. **Stopword Removal**
    5. **Lemmatization**
    6. **TF-IDF Vectorization**
    """)
    
    st.markdown("---")
    st.markdown("Built for NLP Course Project")
    st.markdown("November 2025")

# Check if models are loaded
if not models_loaded:
    st.error("⚠️ **Models not found!**")
    st.markdown("""
    ### How to fix:
    1. Run the export cell at the end of your Colab notebook
    2. Download these files:
       - `logistic_regression_model.pkl`
       - `tfidf_vectorizer.pkl`
    3. Place them in the same directory as `app.py`
    4. Restart the Streamlit app
    """)
    st.stop()

# Main area
st.subheader("📝 Enter Your Movie Review")

# Text input
user_input = st.text_area(
    "Type or paste your review here:",
    height=150,
    placeholder="e.g., This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout..."
)

# Buttons row
col1, col2 = st.columns([4, 1])

with col1:
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

# Clear functionality
if clear_button:
    st.rerun()

# Analysis
if analyze_button:
    if user_input.strip():
        with st.spinner("🤖 Analyzing sentiment..."):
            sentiment, confidence, processed = predict_sentiment(user_input)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Sentiment display
            col1, col2 = st.columns(2)
            
            with col1:
                if "Positive" in sentiment:
                    st.success(f"### {sentiment}")
                else:
                    st.error(f"### {sentiment}")
            
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
            
            # Progress bar
            st.progress(int(confidence)/100)
            
            # Confidence interpretation
            if confidence >= 90:
                st.info("🎯 **Very High Confidence** - The model is very certain about this prediction.")
            elif confidence >= 75:
                st.info("✅ **High Confidence** - The model is confident about this prediction.")
            elif confidence >= 60:
                st.warning("⚠️ **Moderate Confidence** - The model has some uncertainty.")
            else:
                st.warning("❓ **Low Confidence** - The review may have mixed sentiment.")
            
            # Preprocessing details
            with st.expander("🔍 View Preprocessing Steps"):
                st.write("**Step 1: Original Text**")
                st.code(user_input, language=None)
                
                st.write("**Step 2: After HTML Cleaning & Lowercase**")
                step1 = clean_text(user_input)
                st.code(step1, language=None)
                
                st.write("**Step 3: After Stopword Removal**")
                step2 = remove_stopwords(step1)
                st.code(step2, language=None)
                
                st.write("**Step 4: Final (After Lemmatization)**")
                st.code(processed, language=None)
                
                st.info(f"📉 Word count reduced: {len(user_input.split())} → {len(processed.split())} words")
    else:
        st.warning("⚠️ Please enter a review to analyze!")

# Example reviews
st.markdown("---")
st.subheader("💡 Try Example Reviews")

example_reviews = {
    "😊 Positive": "This movie was absolutely fantastic! The acting was superb, the cinematography was breathtaking, and the plot kept me engaged from start to finish. Definitely one of the best films I've seen this year!",
    "😞 Negative": "This was honestly one of the worst movies I've ever seen. The plot was confusing, the acting was terrible, and it felt like a complete waste of time and money. Would not recommend to anyone.",
    "😐 Mixed": "The movie had its moments. Some scenes were really well done and the main actor gave a solid performance, but overall the pacing was slow and the ending felt rushed. It's an okay watch if you have time.",
    "🎭 Another Positive": "Brilliant masterpiece! Outstanding performances from the entire cast. The director did an amazing job bringing this story to life.",
    "👎 Another Negative": "Boring and predictable. Poor writing, weak characters, and terrible dialogue. Save your money and skip this one."
}

# Create columns for example buttons
cols = st.columns(3)

for idx, (label, review) in enumerate(example_reviews.items()):
    with cols[idx % 3]:
        if st.button(label, use_container_width=True, key=f"example_{idx}"):
            st.session_state.selected_example = review

# Display selected example
if 'selected_example' in st.session_state:
    st.text_area("📋 Selected Example (Copy and analyze):", 
                 value=st.session_state.selected_example, 
                 height=120, 
                 key="example_display")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p><strong>Movie Sentiment Analyzer</strong> | NLP Course Project 2025</p>
    <p>Trained on 50,000 IMDB Movie Reviews | Best Model: Logistic Regression (88.59% Accuracy)</p>
    <p>Powered by scikit-learn, NLTK, and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
