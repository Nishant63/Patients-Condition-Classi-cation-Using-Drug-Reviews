# Group 1 P-209 Deployment
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC

df = pd.read_csv("CleanData.csv")

df = df.dropna()
df = df.drop_duplicates()
df["rating"] = df["rating"].astype(int)

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
reviews = vectorizer.fit_transform(df["review"])

model = LogisticRegression(multi_class="ovr")
#model = SVC()
model.fit(reviews, df["condition"])


def predict_condition(review):
    review = vectorizer.transform([review])
    condition = model.predict(review)[0]
    return condition

def recommend_drugs(review):
    condition = predict_condition(review)
    drug_ratings = df[df["condition"] == condition].groupby("drugName")[["rating", "usefulCount"]].mean()
    drug_ratings["rating_usefulCount"] = drug_ratings["rating"] * drug_ratings["usefulCount"]
    recommended_drugs = drug_ratings.nlargest(5, "rating_usefulCount").index.tolist()
    return recommended_drugs

#recommendations
st.title("Patient's Condition Classification Using Drug Reviews")
review = st.text_input("Enter a patient review:")

if st.button("Predict Condition"):
    condition = predict_condition(review)
    st.write("Predicted Condition:", condition)

if st.button("Recommend Drugs"):
    recommended_drugs = recommend_drugs(review)
    st.write("Recommended Drugs:")
    for drug in recommended_drugs:
        st.write("-", drug)


