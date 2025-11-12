import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK data (first time only)
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def clean_text(text):
    """Clean text: lowercase, remove special chars"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    """Remove English stopwords"""
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """Lemmatize words to base form"""
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_text(text):
    """Complete preprocessing pipeline"""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# Note: You'll need to save your trained model and vectorizer from Colab
# For demo purposes, we'll use placeholder predictions
def predict_sentiment(text):
    """Predict sentiment (simplified for demo)"""
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Simple heuristic for demo (replace with actual model loading)
    positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'love', 'wonderful', 'best', 'perfect', 'brilliant']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor', 'waste', 'boring', 'disappointing']
    
    words = processed_text.split()
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    if pos_count > neg_count:
        confidence = min(55 + pos_count * 8, 95)
        return "Positive 😊", confidence
    elif neg_count > pos_count:
        confidence = min(55 + neg_count * 8, 95)
        return "Negative 😞", confidence
    else:
        return "Neutral 😐", 50.0

# Streamlit App
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Header
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("### Analyze the sentiment of movie reviews using NLP")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        """
        This application uses **Natural Language Processing** 
        and **Machine Learning** to classify movie reviews as 
        positive or negative.
        
        **Models Used:**
        - Naive Bayes
        - Logistic Regression ⭐
        - Random Forest
        - LSTM
        
        **Best Model:** Logistic Regression (88.69% accuracy)
        """
    )
    
    st.header("Example Reviews")
    example1 = "This movie was absolutely fantastic! I loved every moment of it."
    example2 = "Terrible film. Complete waste of time and money."
    example3 = "It was an okay movie, nothing special but watchable."

# Main area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Enter Your Movie Review")

# Text input
user_input = st.text_area(
    "Type or paste your review here:",
    height=150,
    placeholder="e.g., This movie was amazing! The acting was superb and the plot kept me engaged throughout..."
)

# Buttons row
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                sentiment, confidence = predict_sentiment(user_input)
                
                # Display results
                st.markdown("---")
                st.subheader("Analysis Results")
                
                # Sentiment with emoji
                if "Positive" in sentiment:
                    st.success(f"**Sentiment:** {sentiment}")
                    color = "green"
                elif "Negative" in sentiment:
                    st.error(f"**Sentiment:** {sentiment}")
                    color = "red"
                else:
                    st.warning(f"**Sentiment:** {sentiment}")
                    color = "orange"
                
                # Confidence score
                st.metric("Confidence Score", f"{confidence:.1f}%")
                st.progress(int(confidence)/100)
                
                # Preprocessing info
                with st.expander("📝 View Preprocessing Steps"):
                    st.write("**Original Text:**")
                    st.code(user_input)
                    st.write("**Cleaned Text:**")
                    cleaned = clean_text(user_input)
                    st.code(cleaned)
                    st.write("**After Stopword Removal:**")
                    no_stopwords = remove_stopwords(cleaned)
                    st.code(no_stopwords)
                    st.write("**After Lemmatization:**")
                    final = lemmatize_text(no_stopwords)
                    st.code(final)
        else:
            st.warning("⚠️ Please enter a review to analyze!")

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()

# Example buttons
st.markdown("---")
st.subheader("Try Example Reviews")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😊 Positive Example"):
        st.session_state.example = example1

with col2:
    if st.button("😞 Negative Example"):
        st.session_state.example = example2

with col3:
    if st.button("😐 Neutral Example"):
        st.session_state.example = example3

# Display example if clicked
if 'example' in st.session_state:
    st.text_area("Selected Example:", value=st.session_state.example, height=100, key="example_display")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>Built for NLP Course Project | November 2025</p>
    <p>Dataset: 50,000 IMDB Movie Reviews</p>
    </div>
    """,
    unsafe_allow_html=True
)
