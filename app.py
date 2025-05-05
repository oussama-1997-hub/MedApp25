import streamlit as st

st.title("Streamlit in Colab")
st.write("Hello from Google Colab!")

name = st.text_input("What's your name?")
if name:
    st.write(f"Hello, {name}!")
