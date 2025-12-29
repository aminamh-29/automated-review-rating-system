#***********************************************************************************************************************************
#IMPORT LIBRARIRES
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

#***************LOADING MODELS AND TF-IDF*********************************************************
model_A=joblib.load("Model_A.pkl")
print("Model_A loaded successfully!")
model_B = joblib.load("Model_B.pkl")
print("Model_B loaded successfully!")
tfidf_B = joblib.load("tfidf_B.pkl")
print("TF-IDF_B loaded successfully!")
#**************AUTOMATED REVIEW RATING SYSTEM**************************************
st.title("Automated Review Rating System")
review = st.text_area("Enter Review:")
if st.button("Submit"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        review_tfidf = tfidf_B.transform([review])

    
        pred_A = model_A.predict(review_tfidf)[0]
        pred_B = model_B.predict(review_tfidf)[0]

        st.subheader("PREDICTIONS")
        st.write("Balanced Model :", pred_A)
        st.write("Imbalanced Model :", pred_B)
