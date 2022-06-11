import streamlit as st
import io

from newspaper import Article

def is_fake(title):
    return True

st.title('kadmus dev #5 application')
text = ""


option = st.selectbox(
        "Choose way to upload article",
        ("Text input", "Upload text file", "Article URL"))

if option == "Text input":
    text = st.text_area("Article text")
elif option == "Upload text file":
    upfile = st.file_uploader("Choose .txt file")
    if upfile is not None:
        stringio = io.StringIO(upfile.getvalue().decode('utf-8'))
        text = stringio.read()
elif option == "Article URL":
    url = st.text_input("Article URL")
    if len(url) > 0:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

if st.button('Accept'):
    if is_fake(text):
        st.subheader("Fake")
    else:
        st.subheader("Not fake")

    st.markdown("**Original text:**")
    st.write(text)
