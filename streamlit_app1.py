import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from nltk.corpus import stopwords
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go

import subprocess
import sys

# Function to download spaCy model
def download_spacy_model():
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Check if the model is already downloaded
try:
    import en_core_web_sm
except ImportError:
    download_spacy_model()

# Download necessary NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the sentiment intensity analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Set the page configuration to wide
st.set_page_config(layout="wide")

# Streamlit app layout
st.title("Sentiment Analysis and Topic Modeling")

# File upload section side by side
col1, col2 = st.columns(2)
with col1:
    topics_file = st.file_uploader("Upload your database file (.xlsx)", type="xlsx")
with col2:
    comments_file = st.file_uploader("Upload your input file (.xlsx)", type="xlsx")

if topics_file and comments_file:
    topics_df = pd.read_excel(topics_file)
    comments_df = pd.read_excel(comments_file)

    if 'comment' not in comments_df.columns:
        st.error("The 'comment' column is not present in your input Excel file.")
    else:
        comments_df['comment'] = comments_df['comment'].str.lower().fillna('')

        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'use'])

        data = [re.sub(r"\'", "", re.sub(r'\s+', ' ', sent)) for sent in comments_df['comment'].tolist()]

        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

        data_words = list(sent_to_words(data))

        bigram = Phrases(data_words, min_count=1, threshold=10)
        trigram = Phrases(bigram[data_words], threshold=10)

        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        def remove_stopwords(texts):
            return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            texts_out = []
            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            for sent in texts:
                doc = nlp(" ".join(sent))
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

        data_words_nostops = remove_stopwords(data_words)
        data_words_bigrams = make_bigrams(data_words_nostops)
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        topic_dict = {}
        for index, row in topics_df.iterrows():
            topic = row['Sub Category']
            word = row['Relevant Word']
            if topic in topic_dict:
                topic_dict[topic].append(word)
            else:
                topic_dict[topic] = [word]

        all_words = [word for words in topic_dict.values() for word in words]

        vectorizer = CountVectorizer(vocabulary=all_words, ngram_range=(1, 3))

        labels = list(topic_dict.keys())

        X_train = vectorizer.transform([' '.join(words) for words in topic_dict.values()])

        model = MultinomialNB()
        model.fit(X_train, range(len(labels)))

        def predict_topic(comment):
            if not comment.strip():
                return 'undefined'
            transformed_comment = vectorizer.transform([comment])
            prediction = model.predict(transformed_comment)
            return labels[prediction[0]] if transformed_comment.sum() else 'undefined'

        # Perform sentiment analysis and topic prediction on each comment
        results = []
        for comment in comments_df['comment']:
            # Topic prediction
            predicted_topic = predict_topic(comment)

            # Vader sentiment analysis
            vader_sentiment_scores = vader_analyzer.polarity_scores(comment)

            # TextBlob sentiment analysis
            blob = TextBlob(comment)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Combine results
            results.append({
                'Comment': comment,
                'Predicted Topic': predicted_topic,
                'Vader_Compound': vader_sentiment_scores['compound'],
                'Vader_Positive': vader_sentiment_scores['pos'],
                'Vader_Neutral': vader_sentiment_scores['neu'],
                'Vader_Negative': vader_sentiment_scores['neg'],
                'TextBlob_Polarity': polarity,
                'TextBlob_Subjectivity': subjectivity
            })

        # Convert the results to a DataFrame
        combined_df = pd.DataFrame(results)

        # Define sentiment categories based on the conditions
        def categorize_sentiment(row):
            vader_compound = row['Vader_Compound']
            vader_positive = row['Vader_Positive']
            vader_neutral = row['Vader_Neutral']
            vader_negative = row['Vader_Negative']
            textblob_polarity = row['TextBlob_Polarity']
            textblob_subjectivity = row['TextBlob_Subjectivity']

            # Negative
            if vader_compound < 0:
                return 'Negative'

            # Neutral
            if vader_compound == 0:
                return 'Neutral'
            
            # Positive
            if vader_compound > 0:
                return 'Positive'

        # Apply the categorization function to each row
        combined_df['Sentiment_Category'] = combined_df.apply(categorize_sentiment, axis=1)

        # Visualization of top mentioned topics
        topic_counts = combined_df['Predicted Topic'].value_counts()
        topic_sentiment_counts = combined_df.groupby(['Predicted Topic', 'Sentiment_Category']).size().unstack(fill_value=0)

        # Convert the data to a format suitable for Plotly
        topic_sentiment_counts = topic_sentiment_counts.reset_index().melt(id_vars='Predicted Topic', var_name='Sentiment', value_name='Count')

        # Horizontal bar chart for top mentioned topics with sentiment proportions
        figA = px.bar(
            topic_sentiment_counts,
            x='Count',
            y='Predicted Topic',
            color='Sentiment',
            orientation='h',
            title='Top Mentioned Topics with Sentiment Proportion',
            labels={'Count': 'Number of Mentions', 'Predicted Topic': 'Topics'},
            color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
        )

        # Update layout for better aesthetics
        figA.update_layout(
            barmode='stack',
            xaxis_title='Number of Mentions',
            yaxis_title='Topics',
            legend_title='Sentiment',
            template='plotly_white',
            margin=dict(l=10, r=10, t=30, b=10)
        )

        # Display the chart in Streamlit
        st.plotly_chart(figA)


        # Visualization of sentiment distribution
        sentiment_distribution = combined_df['Sentiment_Category'].value_counts()
        sentiment_topic_counts = combined_df.groupby(['Sentiment_Category', 'Predicted Topic']).size().unstack(fill_value=0)

        # Convert the data to a format suitable for Plotly
        sentiment_topic_counts = sentiment_topic_counts.reset_index().melt(id_vars='Sentiment_Category', var_name='Topic', value_name='Mentions')
        
        # Bar chart for sentiment distribution with topic proportions
        figB = px.bar(
            sentiment_topic_counts,
            x='Sentiment_Category',
            y='Mentions',
            color='Topic',
            title='Sentiment Distribution with Topic Proportion',
            labels={'Mentions': 'Number of Mentions', 'Sentiment_Category': 'Sentiment'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        # Update layout for better aesthetics
        figB.update_layout(
            barmode='stack',
            xaxis_title='Sentiment',
            yaxis_title='Number of Mentions',
            legend_title='Topic',
            template='plotly_white'
        )

        # Display the chart in Streamlit
        st.plotly_chart(figB)

        # Function to get n-grams and their counts
        def get_ngrams(tokens, n):
            n_grams = ngrams(tokens, n)
            return Counter(n_grams)

        # Tokenize the text
        text_data = combined_df['Comment'].dropna().tolist()
        text = ' '.join(text_data)
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
#
        # Function to get n-grams and their counts
        def get_ngrams(tokens, n):
            n_grams = ngrams(tokens, n)
            return Counter(n_grams)

        # Function to plot n-grams using Plotly
        def plot_ngrams(ngrams, title):
            ngram_df = pd.DataFrame(ngrams.most_common(10), columns=['Ngram', 'Count'])
            ngram_df['Ngram'] = ngram_df['Ngram'].apply(lambda x: ' '.join(x))
            fig = px.bar(ngram_df, x='Count', y='Ngram', orientation='h', title=title, text='Count',color='Ngram',color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(
                xaxis_title='Count',
                yaxis_title='N-gram',
                template='plotly_white'
            )
            return fig

        # Assuming text_data is already defined and contains the text data
        text_data = combined_df['Comment'].dropna().tolist()
        text = ' '.join(text_data)
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

        # Get unigrams, bigrams, trigrams, and fourgrams
        unigrams = get_ngrams(tokens, 1)
        bigrams = get_ngrams(tokens, 2)
        trigrams = get_ngrams(tokens, 3)
        fourgrams = get_ngrams(tokens, 4)

        # Streamlit app layout
        st.title("N-gram Analysis")

        # Add a selectbox for choosing the type of n-gram to display
        ngram_type = st.selectbox("Select N-gram Type", options=["Unigram", "Bigram", "Trigram", "Fourgram"])

        # Display the corresponding graph based on the selected n-gram type
        if ngram_type == "Unigram":
            fig_unigrams = plot_ngrams(unigrams, 'Most Common Unigrams')
            st.plotly_chart(fig_unigrams)
        elif ngram_type == "Bigram":
            fig_bigrams = plot_ngrams(bigrams, 'Most Common Bigrams')
            st.plotly_chart(fig_bigrams)
        elif ngram_type == "Trigram":
            fig_trigrams = plot_ngrams(trigrams, 'Most Common Trigrams')
            st.plotly_chart(fig_trigrams)
        elif ngram_type == "Fourgram":
            fig_fourgrams = plot_ngrams(fourgrams, 'Most Common Fourgrams')
            st.plotly_chart(fig_fourgrams)
#    
        # Function to filter the DataFrame based on sentiment
        def filter_by_sentiment(sentiment):
            return combined_df[combined_df['Sentiment_Category'] == sentiment]

        # Add interactivity to the bar charts
        st.markdown("### Click on the bars to filter the table by sentiment")
        sentiment_filter = st.selectbox("Filter by Sentiment", options=["All", "Positive", "Neutral", "Negative"])
        if sentiment_filter != "All":
            filtered_df = filter_by_sentiment(sentiment_filter)
        else:
            filtered_df = combined_df

        st.dataframe(filtered_df[['Comment', 'Predicted Topic', 'Sentiment_Category']])

        # Provide option for filtered data download
        st.download_button(
            label="Download data as CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='filtered_sentiment_analysis_results.csv',
            mime='text/csv',
        )
        
