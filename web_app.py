import streamlit as st
import io
import subprocess
import sys

from newspaper import Article
from PIL import Image

from detection_module import pipeline_factory

import nltk
nltk.download('punkt')

subprocess.run([f"/bin/bash", "deploy_init.sh"])

fake_probability = pipeline_factory()

col1, col2, col3 = st.columns([4, 6, 1])
with col1:
    st.write("")
with col2:
    logo = Image.open('source_logo.png')
    st.image(logo, output_format='PNG', width=150)
with col3:
    st.write("")

st.markdown("<h1 style='text-align: center;'>kadmus dev #5 application</h1>", unsafe_allow_html=True)
st.markdown("***")
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
    try:
        fake_proba, threshold, matches = fake_probability(text)

        result = "Not fake"
        if fake_proba >= threshold:
            result = "Fake"

        fake_proba *= 100
        verdict = "{} ({}%)".format(result, str(fake_proba))
        st.subheader(verdict)

        src_header, orig_header = st.columns(2)
        with src_header:
            st.subheader("Source Text")
        with orig_header:
            st.subheader("Your Text")

        st.markdown("""---""")
        for m in matches:
            col_src, col_orig = st.columns(2)

            color = 'Green'
            if not m.matched:
                color = 'Gray'
            elif m.is_fake:
                color = 'Red'
            else:
                color = 'Green'

            with col_src:
                if m.source is not None:
                    output = '<p style="color:{};">{}</p>'.format(color, m.source.text)
                    st.markdown(output, unsafe_allow_html=True)
            with col_orig:
                if m.target is not None:
                    output = '<p style="color:{};">{}</p>'.format(color, m.target.text)
                    st.markdown(output, unsafe_allow_html=True)
            st.markdown("""---""")
    except:
        st.markdown("**The source text have not been found**")
        st.subheader("Your text:")
        st.write(text)

#fake_chunks = OriginComparisonResults(
#    matches=(
#        ScoredMatchingResult(
#            source=TextChunk(
#                relative_position=0,
#                text_position=0,
#                text='fdsafjasdkl;fjklasdjdfklasdjfkldjsafkljdfjjfjfjfjfjfjfjjjjkfj\njfjfjfjfjfjfjfjf\nfjfjfjfjf\njfjfjfjf\nfjfjfjfjfjfjf'
#            ),
#            target=TextChunk(
#                relative_position=0,
#                text_position=0,
#                text='fjdfkdjsfjksdjfkdjfkjsdkfjds'
#            ),
#            matched = True,
#            is_fake = True,
#            fact_score = None
#        ),
#        ScoredMatchingResult(
#            source=TextChunk(
#                relative_position=1,
#                text_position=1,
#                text='2dfjfkjdfkjdfjdf\nkjfdkdfkdfjdkf\nkjfkdjfkdjfkd\n'
#            ),
#            target=None,
#            matched=False,
#            fact_score = None,
#            is_fake = None
#        ),
#        ScoredMatchingResult(
#            source=TextChunk(
#                relative_position=2,
#                text_position=2,
#                text='fdkjsfkldsjfkjsdklfjsdklfjklsdf'
#            ),
#            target=TextChunk(
#                relative_position=1,
#                text_position=1,
#                text='fdfasdfdasfsafa\ndsafsafa'
#            ),
#            matched = True,
#            is_fake = False,
#            fact_score = None
#        ),
#        ScoredMatchingResult(
#            source=None,
#            target=TextChunk(
#                relative_position=2,
#                text_position=2,
#                text='fdafasdfsa\nsdafasf\nadfasdfdas\ndfasf'
#            ),
#            matched = False,
#            fact_score = None,
#            is_fake = None
#        )
#    ),
#    matched_proportion=0.7,
#    features = None
#)

