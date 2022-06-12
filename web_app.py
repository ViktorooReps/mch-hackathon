import streamlit as st
import io

from newspaper import Article

from feature_extraction.sequence_matcher.semantic import MatchingResult, TextChunk
from feature_extraction.text_pairs import OriginComparisonFeatures

fake_chunks = OriginComparisonFeatures(
    matches=(
        MatchingResult(
            source=TextChunk(
                relative_position=0,
                text_position=0,
                text='fdsafjasdkl;fjklasdjdfklasdjfkldjsafkljdfjjfjfjfjfjfjfjjjjkfj\njfjfjfjfjfjfjfjf\nfjfjfjfjf\njfjfjfjf\nfjfjfjfjfjfjf'
            ),
            target=TextChunk(
                relative_position=0,
                text_position=0,
                text='fjdfkdjsfjksdjfkdjfkjsdkfjds'
            )
        ),
        MatchingResult(
            source=TextChunk(
                relative_position=1,
                text_position=1,
                text='2dfjfkjdfkjdfjdf\nkjfdkdfkdfjdkf\nkjfkdjfkdjfkd\n'
            ),
            target=None
        ),
        MatchingResult(
            source=TextChunk(
                relative_position=2,
                text_position=2,
                text='fdkjsfkldsjfkjsdklfjsdklfjklsdf'
            ),
            target=TextChunk(
                relative_position=1,
                text_position=1,
                text='2'
            )
        ),
        MatchingResult(
            source=None,
            target=TextChunk(
                relative_position=2,
                text_position=2,
                text='3'
            )
        )
    ),
    fact_scores=(1.0, 0.2),
    matched_proportion=0.7
)


def fake_probability(text):
    return 1, 0.5, fake_chunks.matches


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
            st.subheader("Source text")
        with orig_header:
            st.subheader("Your text")

        for m in matches:
            col_src, col_orig = st.columns(2)

            color = 'Green'
            if m.source is None or m.target is None:
                color = 'Gray'

            with col_src:
                if m.source is not None:
                    output = '<p style="color:{};">{}</p>'.format(color, m.source.text)
                    st.markdown(output, unsafe_allow_html=True)
            with col_orig:
                if m.target is not None:
                    output = '<p style="color:{};">{}</p>'.format(color, m.target.text)
                    st.markdown(output, unsafe_allow_html=True)
    except:
        st.markdown("**The source text have not been found**")
        st.subheader("Your text:")
        st.write(text)

    #if matches is not None:
    #    col_src, col_orig = st.columns(2)
    #    with col_src:
    #        st.subheader("Source text")
    #        #st.write(src_text)
    #        for m in matches:
    #            if m.source is not None:
    #                output = '<p style="color:Green;">{}</p>'.format(m.source.text)
    #                st.markdown(output, unsafe_allow_html=True)
    #    with col_orig:
    #        st.subheader("Your text")
    #        for m in matches:
    #            if m.target is not None:
    #                output = '<p style="color:Green;">{}</p>'.format(m.target.text)
    #                st.markdown(output, unsafe_allow_html=True)
    #else:
    #    st.markdown("**The source text have not been found**")
    #    st.subheader("Your text:")
    #    st.write(text)
    #if is_fake(text):
    #    st.subheader("Fake")
    #else:
    #    st.subheader("Not fake")

    #st.markdown("**Original text:**")
    #st.write(text)
