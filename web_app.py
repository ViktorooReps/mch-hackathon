import streamlit as st
import io

from newspaper import Article

def fake_probability(text):
    return text, 1, 0.5

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
    src_text, fake_proba, threshold = fake_probability(text)

    result = "Not fake"
    if fake_proba >= threshold:
        result = "Fake"

    fake_proba *= 100
    verdict = "{} ({}%)".format(result, str(fake_proba))
    st.subheader(verdict)

    if src_text is not None:
        col_src, col_orig = st.columns(2)
        with col_src:
            st.subheader("Source text")
            st.write(src_text)
        with col_orig:
            st.subheader("Your text")
            st.write(text)
    else:
        st.markdown("**The source text have not been found**")
        st.subheader("Your text:")
        st.write(text)
    #if is_fake(text):
    #    st.subheader("Fake")
    #else:
    #    st.subheader("Not fake")

    #st.markdown("**Original text:**")
    #st.write(text)
