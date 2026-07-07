import streamlit as st
from api import predict

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Sentiment Analysis System")
st.write("Enter your text below and detect sentiment")

user_input = st.text_area("Enter your sentence")

if st.button("Analyze"):
    if user_input.strip() != "":
        result = predict(user_input)

        sentiment = result[0]
        image_path = result[1]

        st.subheader("Prediction Result")
        st.success(sentiment)

        st.image(image_path, width=150)

    else:
        st.warning("Please enter some text")
