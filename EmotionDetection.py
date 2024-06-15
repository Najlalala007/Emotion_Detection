import streamlit as st
import joblib

# Load the model, vectorizer, and label encoder
model = joblib.load('emotion_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit UI
st.title('Emotion Detection')
st.write('Enter a text message to classify its emotion.')

st.write("""
This app predicts the emotion of a given text message using Naive Bayes Classifier Method.
""")

# User input text
user_input = st.text_area('Enter text message here:')

# Button to trigger classification
if st.button('Classify'):
    if user_input.strip() == "":
        st.write('Please enter a valid text message.')
    else:
        # Preprocess and transform input text
        processed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([processed_input])

        # Make prediction
        prediction = model.predict(vectorized_input)
        predicted_emotion = inverse_emotion_mapping[prediction[0]]

        st.write('Predicted emotion:', predicted_emotion)
