import streamlit as st
import streamlit.components.v1 as stc

# Summarization pkgs
# Text Rank algorithm
from gensim.summarization import summarize
 # krävs gensim==3.8.3

# LexRank algorithm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# EDA packages
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # TkAgg # Backend

import neattext.functions as nfx

# Evaluate summary
# from rouge import Rouge

from wordcloud import WordCloud 

# Fxn for LexRank Summarization
def sumy_summarizer(docx, num=3):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, num)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


HTML_BANNER = """
    <div style="background-color:#3872fb;padding:10px;border-radius:10px;border-style:ridge;">
    <h1 style="color:white;text-align:center;">Text Analysis NLP App </h1>
    </div>
    """

def plot_wordcloud(docx):
	mywordcloud = WordCloud().generate(docx)
	fig = plt.figure(figsize=(20,10))
	plt.imshow(mywordcloud,interpolation='bilinear')
	plt.axis('off')
	st.pyplot(fig)

def main():
      
    stc.html(HTML_BANNER)
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Summarization")
        raw_text = st.text_area("Enter TEXT here")
        
        if st.button("Summarize"):
            process_text = nfx.remove_stopwords(raw_text)

            with st.expander("Original Text"):
                st.write(raw_text)
            # layout
            c1,c2 = st.columns(2)
            with c1:
                with st.expander("LexRank Summary"):
                    my_summary = sumy_summarizer(raw_text)
                    document_len = {"Original": len(raw_text), "Summary": len(my_summary)}
                    st.write(document_len)
                    st.write(my_summary)

                with st.expander("Plot Wordcloud"):
                    st.info("Word Cloud")
                    plot_wordcloud(process_text)   


            with c2:
                with st.expander("TextRank Summary"):
                    my_summary = summarize(raw_text)
                    document_len = {"Original": len(raw_text), "Summary": len(my_summary)}
                    st.write(document_len)
                    st.write(my_summary)
        

                    


    else:
        st.subheader("About")
        st.info('The application yields summary of English text by using 2 different technologies and produced a fancy Word Cloud graph')



if __name__ == '__main__':
    main()