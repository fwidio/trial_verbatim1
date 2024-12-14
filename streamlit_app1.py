import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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
from mtranslate import translate # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import plotly.express as px
import concurrent.futures


# Set the page configuration to wide
st.set_page_config(
    layout="wide")

st.markdown("""
    <style>
    div.stButton > button:hover {
        background-color: #428dff;
        color: white;
        border: 2px solid #428dff;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to display the home page
def home():
    st.title(":blue[📌 Welcome to the Python-Based Verbatim Tool]")
    st.caption("Version updated 25 October 2024")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("How to Use.png", use_container_width=True)
        
    with col2:
        if st.button("CSAT | Lifecycle | Pulse"):
            st.session_state.page = "CSAT, Lifecycle, Pulse, Townhall"
        if st.button("Contact Center"):
            st.session_state.page = "Contact Center"
        if st.button("Townhall"):
            st.session_state.page = "Townhall"
        st.caption(":red[_Note: Please double click the button_]")

# Function to display the CSAT, Lifecycle, Pulse, Townhall page
def csat_lifecycle_pulse_townhall():
    # Streamlit app layout
    st.title(":blue[💡Sentiment Analysis and Topic Prediction]")

    # Download necessary NLTK data
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    # Initialize the sentiment intensity analyzer
    vader_analyzer = SentimentIntensityAnalyzer()

    # Custom CSS for the information icon and tooltip
    st.markdown("""
    <style>
    .info-icon {
        display: inline-block;
        position: relative;
        cursor: pointer;
        color: green;
        margin-left: 5px;
        margin-top: 3px; /* Adjust this value to move the icon down */
    }
    .info-icon:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    .tooltip {
        visibility: hidden;
        opacity: 0;
        width: 400px;
        background-color: green;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position the tooltip above the icon */
        left: 50%;
        margin-left: -100px;
        transition: opacity 0.3s;
    }
    .tooltip::after {
        content: "";
        position: absolute;
        top: 100%; /* Arrow at the bottom of the tooltip */
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: black transparent transparent transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    # Language selection with information icon
    st.markdown("""
    <div style="display: flex; align-items: center;">
      <label for="language-select">Select Language</label>
      <div class="info-icon">
          ⓘ
          <span class="tooltip">Default: English 
          If you choose other than English, the machine will automatically convert it to English prior to further processing.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    language = st.selectbox("", ["English", "Indonesia"], key="language-select")

    # File upload section side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Selection of type of data
        data_type = st.selectbox("Select the type of data", ["Pulse Survey", "CSAT Feedback", "Lifecycle","Townhall"])
    
    with col2:
        master_db_path = st.file_uploader("Upload your Master Database (.xlsx)", type="xlsx")
        comments_file = st.file_uploader("Upload your input file (.xlsx)", type="xlsx")
    
    database_path = None
    input_file_path = None

    # Check if both files are uploaded
    if not master_db_path or not comments_file:
        st.error("Both the Master Database and the input file must be uploaded.")
    
    else:
        # Proceed with the rest of your code if both files are uploaded
        if comments_file and master_db_path:
            # Read the appropriate sheet based on the selected data type
            if data_type == "Pulse Survey":
                subtopics_df = pd.read_excel(master_db_path, sheet_name="Pulse")
            elif data_type == "CSAT Feedback":
                subtopics_df = pd.read_excel(master_db_path, sheet_name="CSAT")
            elif data_type == "Lifecycle":
                subtopics_df = pd.read_excel(master_db_path, sheet_name="Lifecycle")
            elif data_type == "Contact Center":
                subtopics_df = pd.read_excel(master_db_path, sheet_name="Contact Center")
            
            comments_df = pd.read_excel(comments_file)
            
            # Translate the comments if the selected language is Indonesian and store in session state
            if language == "Indonesia":
                def translate_comment(comment):
                    try:
                        return translate(comment, 'en', 'id')
                    except Exception as e:
                        st.error(f"Translation error: {e}")
                        return comment
                
                if 'translated_comments' not in st.session_state:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_translation = {executor.submit(translate_comment, comment): i for i, comment in enumerate(comments_df['comment'])}
                        translated_list = [None] * len(comments_df) # Initialize list with None values
                        progress_bar = st.progress(0)
                        status_texts = ["Good things come to those who wait… and wait… and wait…🎉", "Time flies like an arrow; file uploads, not so much..", "Almost there… just a few more bytes to go",
                                        "Patience level: expert", "Loading… because good things take time", "Why rush? Enjoy this moment of zen",
                                        "Just a moment… or two… or three…", "They say patience is a virtue. Consider yourself very virtuous", "This is a test of your patience. You’re doing great!"]
                        status_text_placeholder = st.empty()
                        
                        for i, future in enumerate(concurrent.futures.as_completed(future_to_translation)):
                            index = future_to_translation[future]
                            try:
                                translated_list[index] = future.result()
                            except Exception as e:
                                translated_list[index] = 'none'
                            # Update progress bar
                            progress_bar.progress((i + 1) / len(comments_df))
                            # Update status text less frequently
                            if i % 60 == 0:
                                status_text_placeholder.text(status_texts[(i // 60) % len(status_texts)])
                        # Show completion message
                        status_text_placeholder.text("Processing Complete! You may see the insight now")
                    
                    st.session_state.translated_comments = translated_list
                
                comments_df['comment'] = st.session_state.translated_comments
            
            # Read custom lexicon from the master database
            custom_lexicon_df = pd.read_excel(master_db_path, sheet_name="custom lexicon")
            custom_lexicon = pd.Series(custom_lexicon_df['Sentiment Score'].values, index=custom_lexicon_df['Word']).to_dict()
            
            # Update VADER's lexicon with custom words
            vader_analyzer.lexicon.update(custom_lexicon)
            
            def clean_dataframe(df):
                # Remove rows that are empty or consist only of '-', '_', '0', 'none', or 'no'
                df.replace(['-', '_', '0', 'none', 'no','n/a','.',' ','`-','n.a.','..',"'", "….", "…"], pd.NA, inplace=True)
                df.dropna(how='all', inplace=True)
                return df
            
            # Clean the comments dataframe
            comments_df = clean_dataframe(comments_df)
            
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
                    lemmatizer = WordNetLemmatizer()
                    texts_out = []
                    for sent in texts:
                        texts_out.append([lemmatizer.lemmatize(word) for word in sent if word in allowed_postags])
                    return texts_out
                
                data_words_nostops = remove_stopwords(data_words)
                data_words_bigrams = make_bigrams(data_words_nostops)
                data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
                
                subtopic_dict = {}
                for index, row in subtopics_df.iterrows():
                    subtopic = row['Sub Category']
                    word = row['Relevant Word']
                    if subtopic in subtopic_dict:
                        subtopic_dict[subtopic].append(word)
                    else:
                        subtopic_dict[subtopic] = [word]
                
                all_words = [word for words in subtopic_dict.values() for word in words]
                vectorizer = CountVectorizer(vocabulary=all_words, ngram_range=(1, 3))
                labels = list(subtopic_dict.keys())
                X_train = vectorizer.transform([' '.join(words) for words in subtopic_dict.values()])
                model = MultinomialNB()
                model.fit(X_train, range(len(labels)))
                
                def predict_subtopic(comment):
                    if not comment.strip():
                        return 'undefined'
                    if comment == "there isn't any":
                        return 'Not Applicable'
                    transformed_comment = vectorizer.transform([comment])
                    prediction = model.predict(transformed_comment)
                    return labels[prediction[0]] if transformed_comment.sum() else 'undefined'
                
                # Perform sentiment analysis and subtopic prediction on each comment
                results = []
                for comment in comments_df['comment']:
                    # SubTopic prediction
                    predicted_subtopic = predict_subtopic(comment)
                    # Vader sentiment analysis
                    vader_sentiment_scores = vader_analyzer.polarity_scores(comment)
                    # TextBlob sentiment analysis
                    blob = TextBlob(comment)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    # Combine results
                    results.append({
                        'Comment': comment,
                        'Predicted Sub Topic': predicted_subtopic,
                        'Vader_Compound': vader_sentiment_scores['compound'],
                        'Vader_Positive': vader_sentiment_scores['pos'],
                        'Vader_Neutral': vader_sentiment_scores['neu'],
                        'Vader_Negative': vader_sentiment_scores['neg'],
                        'TextBlob_Polarity': polarity,
                        'TextBlob_Subjectivity': subjectivity
                    })
                
                # Convert the results to a DataFrame
                combined_df = pd.DataFrame(results)
                
                # Add the category/topic column based on the predicted sub topic
                def get_topic(subtopic):
                    topic = subtopics_df.loc[subtopics_df['Sub Category'] == subtopic, 'Category']
                    return topic.values[0] if not topic.empty else 'undefined'
                
                combined_df['Category'] = combined_df['Predicted Sub Topic'].apply(get_topic)
                # Rename the 'Category' column to 'Topic'
                combined_df = combined_df.rename(columns={'Category': 'Predicted Topic'})
                
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
                    if 0 <= vader_compound <= 0.1:
                        return 'Neutral'
                    # Positive
                    if vader_compound > 0:
                        return 'Positive'
                
                # Apply the categorization function to each row
                combined_df['Sentiment Category'] = combined_df.apply(categorize_sentiment, axis=1)
                
                # Group the data by 'Topic' and 'Sub Topic' to get the counts
                topic_subtopic_counts = combined_df.groupby(['Predicted Topic', 'Predicted Sub Topic']).size().reset_index(name='Count')
                # Sort the data by 'Count' in descending order
                topic_subtopic_counts = topic_subtopic_counts.sort_values(by='Count', ascending=False)
                
                # Create a bar chart
                fig = px.bar(
                    topic_subtopic_counts,
                    x='Predicted Topic',
                    y='Count',
                    color='Predicted Sub Topic',
                    title='Distribution of Sub Topics within Topics',
                    labels={'Predicted Topic': 'Topic', 'Count': 'Number of Mentions', 'Predicted Sub Topic': 'Sub Topic'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                # Update layout for better aesthetics
                fig.update_layout(
                    barmode='stack',
                    xaxis_title='Topic',
                    yaxis_title='Number of Mentions',
                    legend_title='Sub Topic',
                    template='plotly_white'
                )
                # Display the chart in Streamlit
                st.plotly_chart(fig)
                
                # Apply the categorization function to each row
                combined_df['Sentiment Category'] = combined_df.apply(categorize_sentiment, axis=1)
                
                # Add a selector for choosing between 'Topic' and 'Sub Topic'
                attribute_selector = st.selectbox("Select Attribute to Display", options=['Predicted Topic', 'Predicted Sub Topic'])
                
                # Visualization of top mentioned topics or subtopics with sentiment proportions
                attribute_counts = combined_df[attribute_selector].value_counts()
                attribute_sentiment_counts = combined_df.groupby([attribute_selector, 'Sentiment Category']).size().unstack(fill_value=0)
                # Calculate the total number of mentions for each attribute
                attribute_sentiment_counts['Total'] = attribute_sentiment_counts.sum(axis=1)
                # Sort the data by 'Total' in descending order
                attribute_sentiment_counts = attribute_sentiment_counts.sort_values(by='Total', ascending=True)
                # Convert the data to a format suitable for Plotly
                attribute_sentiment_counts = attribute_sentiment_counts.reset_index().melt(id_vars=[attribute_selector, 'Total'], var_name='Sentiment', value_name='Count')
                
                # Horizontal bar chart for top mentioned topics or subtopics with sentiment proportions
                figA = px.bar(
                    attribute_sentiment_counts,
                    x='Count',
                    y=attribute_selector,
                    color='Sentiment',
                    orientation='h',
                    title=f'Top Mentioned {attribute_selector.replace("_", " ")}s with Sentiment Proportion',
                    labels={'Count': 'Number of Mentions', attribute_selector: attribute_selector.replace("_", " ")},
                    color_discrete_map={'Positive': 'green', 'Neutral': 'silver', 'Negative': 'red'}
                )
                # Update layout for better aesthetics
                figA.update_layout(
                    barmode='stack',
                    xaxis_title='Number of Mentions',
                    yaxis_title=attribute_selector.replace("_", " "),
                    legend_title='Sentiment',
                    template='plotly_white'
                )
                # Display the chart in Streamlit
                st.plotly_chart(figA)
                ########################################################
                # Function to get n-grams and their counts
                def get_ngrams(tokens, n):
                    n_grams = ngrams(tokens, n)
                    return Counter(n_grams)

                # Define the preprocess_text function
                def preprocess_text(text):
                    # Replace "this&that" with "this_and_that"
                    text = re.sub(r'(\w+)&(\w+)', r'\1_and_\2', text)
                    # Apply simple_preprocess
                    tokens = gensim.utils.simple_preprocess(text, deacc=True)
                    # Revert the placeholder back to "&"
                    tokens = [token.replace("_and_", "&") for token in tokens]
                    return tokens

                # Function to plot n-grams using Plotly
                def plot_ngrams(ngrams, title):
                    ngram_df = pd.DataFrame(ngrams.most_common(10), columns=['Ngram', 'Count'])
                    ngram_df['Ngram'] = ngram_df['Ngram'].apply(lambda x: ' '.join(x))
                    fig = px.bar(ngram_df, x='Count', y='Ngram', orientation='h', title=title, text='Count', color='Ngram', color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(
                        xaxis_title='Count',
                        yaxis_title='N-gram',
                        template='plotly_white',
                        showlegend=False
                    )
                    return fig

                # Streamlit app layout
                st.markdown("<h1 style='font-size:24px;'>N-gram Analysis</h1>", unsafe_allow_html=True)
                # Add a selectbox for choosing the type of n-gram to display
                ngram_type = st.selectbox("Select N-gram Type", options=["Unigram", "Bigram", "Trigram", "Fourgram"])
                # Add filters for topic and subtopic side by side
                col1, col2 = st.columns(2)
                with col1:
                    selected_topic = st.selectbox("Filter by Topic", options=["All"] + combined_df['Predicted Topic'].unique().tolist())
                with col2:
                    if selected_topic != "All":
                        filtered_subtopics = combined_df[combined_df['Predicted Topic'] == selected_topic]['Predicted Sub Topic'].unique().tolist()
                    else:
                        filtered_subtopics = combined_df['Predicted Sub Topic'].unique().tolist()
                    selected_subtopic = st.selectbox("Filter by Sub Topic", options=["All", "Undefined"] + filtered_subtopics)

                # Filter the data based on the selected topic and subtopic
                if selected_topic != "All":
                    filtered_df = combined_df[combined_df['Predicted Topic'] == selected_topic]
                else:
                    filtered_df = combined_df
                if selected_subtopic != "All":
                    filtered_df = filtered_df[filtered_df['Predicted Sub Topic'] == selected_subtopic]

                # Tokenize the text
                text_data = filtered_df['Comment'].dropna().tolist()
                text = ' '.join(text_data)
                # Preprocess the text
                tokens = preprocess_text(text)
                # Remove stopwords and non-alphanumeric tokens
                stop_words = set(stopwords.words('english'))
                custom_stop_words = {'yet', 'quite', 'pmi', 'never', 'stage', 'took', 'needed'}  # Add custom stop words here
                tokens = [word for word in tokens if word.isalnum() and word not in stop_words and word not in custom_stop_words]

                # Get unigrams, bigrams, trigrams, and fourgrams
                unigrams = get_ngrams(tokens, 1)
                bigrams = get_ngrams(tokens, 2)
                trigrams = get_ngrams(tokens, 3)
                fourgrams = get_ngrams(tokens, 4)

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
                
                ######################
                # Function to filter the DataFrame based on sentiment, sub topic, and topic
                def filter_data(sentiment, subtopic, topic):
                    filtered_df = combined_df
                    if sentiment != "All":
                        filtered_df = filtered_df[filtered_df['Sentiment Category'] == sentiment]
                    if subtopic != "All":
                        filtered_df = filtered_df[filtered_df['Predicted Sub Topic'] == subtopic]
                    if topic != "All":
                        filtered_df = filtered_df[filtered_df['Predicted Topic'] == topic]
                    return filtered_df
                
                # Add interactivity to the bar charts
                st.markdown("### Detailed Verbatim")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_filter = st.selectbox("Filter by Sentiment", options=["All", "Positive", "Neutral", "Negative"])
                
                with col2:
                    subtopic_filter = st.selectbox("Filter by Sub Topic", options=["All", "undefined"] + labels)
                
                with col3:
                    topic_filter = st.selectbox("Filter by Topic", options=["All", "undefined"] + subtopics_df['Category'].unique().tolist())
                
                filtered_df = filter_data(sentiment_filter, subtopic_filter, topic_filter)
                
                # Display the DataFrame with the renamed column
                st.dataframe(filtered_df[['Comment', 'Predicted Sub Topic', 'Predicted Topic', 'Sentiment Category']])
                
                # Provide option for filtered data download
                st.download_button(
                    label="Download data as CSV",
                    data=filtered_df.to_csv(index=False).encode('utf-8'),
                    file_name='filtered_sentiment_analysis_results.csv',
                    mime='text/csv',
                )
                
                if st.button("Back to Home Page"):
                    st.session_state.page = "home"

                    
# Function to display the Contact Center page
def contact_center():
    st.write(":red[_Click below button for higher result accuracy (Note: This process may take longer time_])")
    if st.button("High Performance Processing"):
        st.session_state.page = "High Performance Processing"

    # Function to clean the 'Subject' column
    def clean_subject(subject):
        if pd.isna(subject):
            return ""
        subject = subject.replace('#', '')
        subject = subject.replace('_', ' ')
        subject = subject.replace('**', '')
        subject = subject.replace(':', '')
        subject = subject.replace('/', ' ')
        subject = subject.replace('(', '')
        subject = subject.replace(')', '')
        return subject

    # Function to check if the first word is all uppercase
    def is_first_word_uppercase(sentence):
        words = sentence.split()
        return words[0].isupper()

    # Streamlit app
    st.title(":blue[📞Topic & Sub Topic Finder - Contact Center]")

    # Create two columns
    col1, col2 = st.columns(2)

    # Upload input file in the first column
    with col1:
        input_file = st.file_uploader("Upload Input File", type=["xlsx"])

    # Upload database file in the second column
    with col2:
        database_file = st.file_uploader("Upload Database File", type=["xlsx"])

    if input_file and database_file:
        # Read the input file
        input_df = pd.read_excel(input_file)
        # Read the topic database file
        topic_df = pd.read_excel(database_file, sheet_name='tag')
        # Read the subtopic database file
        subtopic_df = pd.read_excel(database_file, sheet_name='subtag')
        
        # Clean the 'Subject' column in both dataframes
        input_df['Subject'] = input_df['Subject'].apply(clean_subject)
        topic_df['Cleaned_Subject'] = topic_df['Subject'].apply(clean_subject)
        subtopic_df['Cleaned_Subject'] = subtopic_df['Subject'].apply(clean_subject)
        
        # Remove blank rows in the database
        topic_df = topic_df[topic_df['Cleaned_Subject'] != ""]
        subtopic_df = subtopic_df[subtopic_df['Cleaned_Subject'] != ""]
        
        # Initialize columns for 'Topic' and 'Sub Topic'
        input_df['Topic'] = ""
        input_df['Sub Topic'] = ""
        
        # Vectorize the subjects using TF-IDF for topics
        vectorizer_topic = TfidfVectorizer().fit_transform(topic_df['Cleaned_Subject'].tolist() + input_df['Subject'].tolist())
        
        # Calculate cosine similarity between input subjects and topic subjects
        cosine_similarities_topic = cosine_similarity(vectorizer_topic[len(topic_df):], vectorizer_topic[:len(topic_df)])
        
        # Match subjects and assign 'Topic'
        for i, row in input_df.iterrows():
            if is_first_word_uppercase(row['Subject']):
                first_word = row['Subject'].split()[0]
                matched_topic = topic_df[topic_df['Cleaned_Subject'].str.startswith(first_word)]['Topic']
                if not matched_topic.empty:
                    input_df.at[i, 'Topic'] = matched_topic.values[0]
            else:
                most_similar_idx_topic = cosine_similarities_topic[i].argmax()
                if cosine_similarities_topic[i][most_similar_idx_topic] > 0.5:  # You can adjust the threshold as needed
                    input_df.at[i, 'Topic'] = topic_df.iloc[most_similar_idx_topic]['Topic']
        
        # Vectorize the subjects using TF-IDF for subtopics
        vectorizer_subtopic = TfidfVectorizer().fit_transform(subtopic_df['Cleaned_Subject'].tolist() + input_df['Subject'].tolist())
        
        # Calculate cosine similarity between input subjects and subtopic subjects
        cosine_similarities_subtopic = cosine_similarity(vectorizer_subtopic[len(subtopic_df):], vectorizer_subtopic[:len(subtopic_df)])
        
        # Match subjects and assign 'Sub Topic'
        for i, row in input_df.iterrows():
            most_similar_idx_subtopic = cosine_similarities_subtopic[i].argmax()
            if cosine_similarities_subtopic[i][most_similar_idx_subtopic] > 0.5:  # You can adjust the threshold as needed
                input_df.at[i, 'Sub Topic'] = subtopic_df.iloc[most_similar_idx_subtopic]['Sub Topic']
        
        # Handle cases where topic is blank by reading the 'blank' sheet
        blank_sheet = pd.read_excel(database_file, sheet_name='blank')
        
        for i, row in input_df.iterrows():
            if row['Topic'] == "":
                for j, blank_row in blank_sheet.iterrows():
                    relevant_word = blank_row['Relevant Word']
                    if relevant_word == "TA":
                        if "TA" in row['Subject']:
                            input_df.at[i, 'Topic'] = blank_row['Topic']
                            break
                    elif relevant_word.lower() in row['Subject'].lower():
                        input_df.at[i, 'Topic'] = blank_row['Topic']
                        break

        # Display the updated dataframe
        st.write("Updated DataFrame:")
        st.dataframe(input_df)
        
        # Option to download the updated dataframe
        st.download_button(
            label="Download Updated Data",
            data=input_df.to_csv(index=False).encode('utf-8'),
            file_name='updated_input_file.csv',
            mime='text/csv'
        )
        
        # Count the number of occurrences of each topic
        topic_counts = input_df['Topic'].value_counts().sort_values(ascending=False)
        
        # Get top 10 topics with their counts and percentages
        top_10_topics_counts = topic_counts.head(10)
        top_10_topics_percentages = (top_10_topics_counts / topic_counts.sum()) * 100
        
        # Create DataFrame with formatted percentages rounded to one decimal place
        top_10_table_data = pd.DataFrame({
            'Topic': top_10_topics_counts.index,
            'Count': top_10_topics_counts.values,
            'Percentage': [f"{x:.1f}%" for x in top_10_topics_percentages.values]
        })

        # Adjust the index to start from 1
        top_10_table_data.index = range(1, len(top_10_table_data) + 1)

        # Display total number of transactions above the table
        total_transactions = len(input_df)
        st.write(f"Total transactions: {total_transactions}")

        # Display the table dynamically
        st.write("Top 10 Topics with Counts and Percentages:")
        st.dataframe(top_10_table_data.style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        ))
        
        # Get top 3 subtopics for each of the top 10 topics with their counts and percentages compared to overall data
        top_3_subtopics_data_overall = []
        for topic in top_10_topics_counts.index:
            subtopics_counts_overall = input_df[input_df['Topic'] == topic]['Sub Topic'].value_counts().head(3)
            subtopics_percentages_overall = (subtopics_counts_overall / total_transactions) * 100

            for subtopic, count, percentage in zip(subtopics_counts_overall.index, subtopics_counts_overall.values, subtopics_percentages_overall.values):
                top_3_subtopics_data_overall.append({
                    'Topic': topic,
                    'Sub Topic': subtopic,
                    'Count': count,
                    'Percentage': f"{percentage:.1f}%"
                })

        top_3_subtopics_table_data_overall = pd.DataFrame(top_3_subtopics_data_overall)

        # Display the subtopic table dynamically
        st.write("Top 3 Subtopics for Each Top Topic with Counts and Percentages Compared to Overall Data:")
        st.dataframe(top_3_subtopics_table_data_overall.style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        ))

        # Plot the bar chart with Plotly Express for better aesthetics and dynamic interactivity with Streamlit
        st.write("Number of Topics (sorted by highest to lowest count):")
        
        fig1 = px.bar(topic_counts, x=topic_counts.values, y=topic_counts.index, orientation='h', 
                    labels={'x': 'Count', 'y': 'Topics'}, title='Number of Topics')
        
        # Update layout to format percentages, start numbering from 1, and center text
        fig1.update_layout(yaxis={'categoryorder':'total ascending'}, height=1000)
        fig1.update_traces(texttemplate='%{x}', textposition='inside', insidetextanchor='middle')
        
        st.plotly_chart(fig1)

    if st.button("Back to Home Page"):
        st.session_state.page = "home"
        
        
def high_performance_processing():
    st.title(":blue[High Performance Processing]")
    st.write(":red[_Attention: This process may take longer time_]")
        
    # Function to clean the 'Subject' column
    def clean_subject(subject):
        if pd.isna(subject):
            return ""
        subject = subject.replace('#', '')
        subject = subject.replace('_', ' ')
        subject = subject.replace('**', '')
        subject = subject.replace(':', '')
        subject = subject.replace('/', ' ')
        subject = subject.replace('(', '')
        subject = subject.replace(')', '')
        return subject.strip()

    # Function to check if the first word is all uppercase
    def is_first_word_uppercase(sentence):
        words = sentence.split()
        return words[0].isupper()

    # Upload input file
    input_file = st.file_uploader("Upload Input File", type=["xlsx"])

    # Database file path
    database_file_path = r"C:\Users\fwidio\Documents\iCare Tag Mapping.xlsx"

    if input_file:
        # Read the input file
        input_df = pd.read_excel(input_file)
        
        # Ensure the 'Subject' column exists
        if 'Subject' not in input_df.columns:
            st.error("The uploaded file does not contain a 'Subject' column.")
        else:
            # Read the topic database file
            topic_df = pd.read_excel(database_file_path, sheet_name='tag')
            # Read the subtopic database file
            subtopic_df = pd.read_excel(database_file_path, sheet_name='subtag')
            
            # Clean the 'Subject' column in both dataframes
            input_df['Subject'] = input_df['Subject'].apply(clean_subject)
            topic_df['Cleaned_Subject'] = topic_df['Subject'].apply(clean_subject)
            subtopic_df['Cleaned_Subject'] = subtopic_df['Subject'].apply(clean_subject)
            
            # Remove blank rows in the database
            topic_df = topic_df[topic_df['Cleaned_Subject'] != ""]
            subtopic_df = subtopic_df[subtopic_df['Cleaned_Subject'] != ""]
            
            # Initialize columns for 'Topic' and 'Sub Topic'
            input_df['Topic'] = ""
            input_df['Sub Topic'] = ""
            
            # Process each row independently
            for i, row in input_df.iterrows():
                # Vectorize the subjects using TF-IDF for topics
                vectorizer_topic = TfidfVectorizer().fit_transform(topic_df['Cleaned_Subject'].tolist() + [row['Subject']])
                
                # Calculate cosine similarity between input subject and topic subjects
                cosine_similarities_topic = cosine_similarity(vectorizer_topic[-1], vectorizer_topic[:-1])
                
                # Match subject and assign 'Topic'
                if is_first_word_uppercase(row['Subject']):
                    first_word = row['Subject'].split()[0]
                    matched_topic = topic_df[topic_df['Cleaned_Subject'].str.startswith(first_word)]['Topic']
                    if not matched_topic.empty:
                        input_df.at[i, 'Topic'] = matched_topic.values[0]
                else:
                    most_similar_idx_topic = cosine_similarities_topic[0].argmax()
                    if cosine_similarities_topic[0][most_similar_idx_topic] > 0.5:  # You can adjust the threshold as needed
                        input_df.at[i, 'Topic'] = topic_df.iloc[most_similar_idx_topic]['Topic']
                
                # Vectorize the subjects using TF-IDF for subtopics
                vectorizer_subtopic = TfidfVectorizer().fit_transform(subtopic_df['Cleaned_Subject'].tolist() + [row['Subject']])
                
                # Calculate cosine similarity between input subject and subtopic subjects
                cosine_similarities_subtopic = cosine_similarity(vectorizer_subtopic[-1], vectorizer_subtopic[:-1])
                
                # Match subject and assign 'Sub Topic'
                most_similar_idx_subtopic = cosine_similarities_subtopic[0].argmax()
                if cosine_similarities_subtopic[0][most_similar_idx_subtopic] > 0.5:  # You can adjust the threshold as needed
                    input_df.at[i, 'Sub Topic'] = subtopic_df.iloc[most_similar_idx_subtopic]['Sub Topic']
            
            # Handle cases where topic is blank by reading the 'blank' sheet
            blank_sheet = pd.read_excel(database_file_path, sheet_name='blank')
            
            for i, row in input_df.iterrows():
                if row['Topic'] == "":
                    for j, blank_row in blank_sheet.iterrows():
                        relevant_word = blank_row['Relevant Word']
                        if relevant_word == "TA":
                            if "TA" in row['Subject']:
                                input_df.at[i, 'Topic'] = blank_row['Topic']
                                break
                        elif relevant_word.lower() in row['Subject'].lower():
                            input_df.at[i, 'Topic'] = blank_row['Topic']
                            break

            # Display the updated dataframe
            st.write("Updated DataFrame:")
            st.dataframe(input_df)
            
            # Option to download the updated dataframe
            st.download_button(
                label="Download Updated Data",
                data=input_df.to_csv(index=False).encode('utf-8'),
                file_name='updated_input_file.csv',
                mime='text/csv'
            )
            
            # Count the number of occurrences of each topic
            topic_counts = input_df['Topic'].value_counts().sort_values(ascending=False)
            
            # Get top 10 topics with their counts and percentages
            top_10_topics_counts = topic_counts.head(10)
            top_10_topics_percentages = (top_10_topics_counts / topic_counts.sum()) * 100
            
            # Create DataFrame with formatted percentages rounded to one decimal place
            top_10_table_data = pd.DataFrame({
                'Topic': top_10_topics_counts.index,
                'Count': top_10_topics_counts.values,
                'Percentage': [f"{x:.1f}%" for x in top_10_topics_percentages.values]
            })

            # Adjust the index to start from 1
            top_10_table_data.index = range(1, len(top_10_table_data) + 1)

            # Display total number of transactions above the table
            total_transactions = len(input_df)
            st.write(f"Total transactions: {total_transactions}")

            # Display the table dynamically
            st.write("Top 10 Topics with Counts and Percentages:")
            st.dataframe(top_10_table_data.style.set_properties(**{'text-align': 'center'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]}]
            ))
            
            # Get top 3 subtopics for each of the top 10 topics with their counts and percentages compared to overall data
            top_3_subtopics_data_overall = []
            for topic in top_10_topics_counts.index:
                subtopics_counts_overall = input_df[input_df['Topic'] == topic]['Sub Topic'].value_counts().head(3)
                subtopics_percentages_overall = (subtopics_counts_overall / total_transactions) * 100

                for subtopic, count, percentage in zip(subtopics_counts_overall.index, subtopics_counts_overall.values, subtopics_percentages_overall.values):
                    top_3_subtopics_data_overall.append({
                        'Topic': topic,
                        'Sub Topic': subtopic,
                        'Count': count,
                        'Percentage': f"{percentage:.1f}%"
                    })

            top_3_subtopics_table_data_overall = pd.DataFrame(top_3_subtopics_data_overall)

            # Display the subtopic table dynamically
            st.write("Top 3 Subtopics for Each Top Topic with Counts and Percentages Compared to Overall Data:")
            st.dataframe(top_3_subtopics_table_data_overall.style.set_properties(**{'text-align': 'center'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]}]
            ))

            # Plot the bar chart with Plotly Express for better aesthetics and dynamic interactivity with Streamlit
            st.write("Number of Topics (sorted by highest to lowest count):")
            
            fig1 = px.bar(topic_counts, x=topic_counts.values, y=topic_counts.index, orientation='h', 
                        labels={'x': 'Count', 'y': 'Topics'}, title='Number of Topics')
            
            # Update layout to format percentages, start numbering from 1, and center text
            fig1.update_layout(yaxis={'categoryorder':'total ascending'}, height=1000)
            fig1.update_traces(texttemplate='%{x}', textposition='inside', insidetextanchor='middle')
            
            st.plotly_chart(fig1)
            
    col = st.columns(1)
    with col[0]:
        if st.button("Back to Home Page"):
            st.session_state.page = "home"
        if st.button("Back to Contact Center"):
            st.session_state.page = "Contact Center"

def Townhall():
    # Function to clean and translate text for sentiment analysis
    def clean_translate_texts(texts):
        cleaned_texts = []
        for text in texts:
            if pd.isna(text) or str(text).strip() in ['-', '_', '0', 'none', 'no', 'n/a', '.', ' ', '`-', 'n.a.', '..', '…']:
                cleaned_texts.append('none')
            else:
                cleaned_texts.append(str(text))  # Convert to string to avoid TypeError

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                translated_texts = list(executor.map(lambda text: translate(text, 'en'), cleaned_texts))
            return translated_texts
        except Exception as e:
            return cleaned_texts

    # Function to predict sentiment using VADER and TextBlob
    def predict_sentiment(text):
        if text == 'none':
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}, 'none'
        
        vader_scores = vader_analyzer.polarity_scores(text)

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if vader_scores['compound'] > 0.1:
            sentiment = 'Positive'
        elif vader_scores['compound'] < 0 and vader_scores['neg'] <= 0.3:
            sentiment = 'Positive'
        elif vader_scores['compound'] >= 0:
            sentiment = 'Neutral'
        else:
            sentiment = 'Negative'

        return vader_scores, sentiment

    # Function to predict topic and subtopic
    def predict_topic_subtopic(text, vectorizer, model, labels, subtopics_df):
        if pd.isna(text) or str(text).strip() in ['-', '_', '0', 'none', 'no', 'n/a', '.', ' ', '`-', 'n.a.', '..', '…']:
            return ('none', 'none')

        transformed_text = vectorizer.transform([str(text)])
        prediction = model.predict(transformed_text)

        if transformed_text.sum() == 0:
            return ('none', 'none')

        subtopic = labels[prediction[0]]
        topic = subtopics_df.loc[subtopics_df['Sub Category'] == subtopic, 'Category'].values[0]

        return (topic, subtopic)

    # Streamlit app layout
    st.title(":blue[🎪Townhall Analysis]")
    st.write(":gray[1. Ensure you have the following **column title/header** in your **input file** = 'Comment', 'Improvement', 'Next Topic']")
    st.write(":gray[2. Ensure you have the following **sheets name** in your **database file** = 'Townhall_Improvement', 'Townhall_NextTopic']")

    # File upload section
    col1, col2 = st.columns(2)
    with col1:
        master_db_path = st.file_uploader("Upload your Master Database (.xlsx)", type="xlsx")
    with col2:
        comments_file = st.file_uploader("Upload your Input File (.xlsx)", type="xlsx")

    # Check if both files are uploaded
    if not master_db_path or not comments_file:
        st.error("Both the Master Database and the Input File must be uploaded.")
    else:
        # Use session state to store data
        if 'input_df' not in st.session_state or 'master_db' not in st.session_state or st.session_state.master_db_path != master_db_path or st.session_state.comments_file != comments_file:
            # Load the Excel files
            master_db = pd.ExcelFile(master_db_path)
            input_file = pd.ExcelFile(comments_file)

            # Read the sheets into DataFrames
            input_df = pd.read_excel(input_file)

            # Initialize the sentiment intensity analyzer
            vader_analyzer = SentimentIntensityAnalyzer()

            # Load the master database for subtopics for Improvement
            subtopics_improvement_df = pd.read_excel(master_db, sheet_name='Townhall_Improvement')

            # Create a dictionary for subtopics and their relevant words for Improvement
            subtopic_improvement_dict = {}
            for index, row in subtopics_improvement_df.iterrows():
                subtopic = row['Sub Category']
                word = row['Relevant Word']
                if subtopic in subtopic_improvement_dict:
                    subtopic_improvement_dict[subtopic].append(word)
                else:
                    subtopic_improvement_dict[subtopic] = [word]

            all_words_improvement = [word for words in subtopic_improvement_dict.values() for word in words]
            vectorizer_improvement = CountVectorizer(vocabulary=all_words_improvement, ngram_range=(1, 3))
            labels_improvement = list(subtopic_improvement_dict.keys())
            X_train_improvement = vectorizer_improvement.transform([' '.join(words) for words in subtopic_improvement_dict.values()])
            model_improvement = MultinomialNB()
            model_improvement.fit(X_train_improvement, range(len(labels_improvement)))

            # Load the master database for subtopics for Next Topic
            subtopics_next_topic_df = pd.read_excel(master_db, sheet_name='Townhall_NextTopic')

            # Create a dictionary for subtopics and their relevant words for Next Topic
            subtopic_next_topic_dict = {}
            for index, row in subtopics_next_topic_df.iterrows():
                subtopic = row['Sub Category']
                word = row['Relevant Word']
                if subtopic in subtopic_next_topic_dict:
                    subtopic_next_topic_dict[subtopic].append(word)
                else:
                    subtopic_next_topic_dict[subtopic] = [word]

            all_words_next_topic = [word for words in subtopic_next_topic_dict.values() for word in words]
            vectorizer_next_topic = CountVectorizer(vocabulary=all_words_next_topic, ngram_range=(1, 3))
            labels_next_topic = list(subtopic_next_topic_dict.keys())
            X_train_next_topic = vectorizer_next_topic.transform([' '.join(words) for words in subtopic_next_topic_dict.values()])
            model_next_topic = MultinomialNB()
            model_next_topic.fit(X_train_next_topic, range(len(labels_next_topic)))

            # Translate the Comment column for sentiment analysis with progress bar and interactive text
            progress_bar = st.progress(0)
            status_texts = ["Good things come to those who wait… and wait… and wait…🎉", "Time flies like an arrow; file uploads, not so much..", "Almost there… just a few more bytes to go",
                            "Patience level: expert", "Loading… because good things take time", "Why rush? Enjoy this moment of zen",
                            "Just a moment… or two… or three…", "They say patience is a virtue. Consider yourself very virtuous", "This is a test of your patience. You’re doing great!"]
            status_text_placeholder = st.empty()

            # Use ThreadPoolExecutor to speed up translation process
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_translation = {executor.submit(translate, text, 'en'): i for i, text in enumerate(input_df['Comment'])}
                translated_list = [None] * len(input_df)  # Initialize list with None values
                for future in concurrent.futures.as_completed(future_to_translation):
                    i = future_to_translation[future]
                    try:
                        translated_list[i] = future.result()
                    except Exception as e:
                        translated_list[i] = 'none'

                    # Update progress bar
                    progress_bar.progress((i + 1) / len(input_df))

                    # Update status text less frequently
                    if i % 50 == 0:
                        status_text_placeholder.text(status_texts[(i // 50) % len(status_texts)])

            # Show completion message
            status_text_placeholder.text("Processing Complete! You may see the insight now")

            # Custom HTML/CSS for green progress bar
            st.markdown(
                """
                <style>
                .stProgress > div > div > div > div {
                    background-color: green;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Set progress bar to 100% and change color to green
            progress_bar.progress(1.0)

            input_df['Comment_Translated'] = translated_list

            # Process the Comment column for sentiment analysis using translated text
            vader_scores_list = []
            sentiments = []
            for text in input_df['Comment_Translated']:
                vader_scores, sentiment = predict_sentiment(text)
                vader_scores_list.append(vader_scores)
                sentiments.append(sentiment)

            input_df['Comment Sentiment'] = sentiments
            input_df['VADER_neg'] = [score['neg'] for score in vader_scores_list]
            input_df['VADER_neu'] = [score['neu'] for score in vader_scores_list]
            input_df['VADER_pos'] = [score['pos'] for score in vader_scores_list]
            input_df['VADER_compound'] = [score['compound'] for score in vader_scores_list]

            # Process the Improvement column for topic-subtopic prediction using original text
            input_df[['Improvement Topic', 'Improvement Sub Topic']] = input_df.apply(lambda x: pd.Series(predict_topic_subtopic(x['Improvement'], vectorizer_improvement, model_improvement, labels_improvement, subtopics_improvement_df)), axis=1)

            # Process the Next Topic column for topic-subtopic prediction using original text
            input_df[['Next Topic Category', 'Next Topic Sub Category']] = input_df.apply(lambda x: pd.Series(predict_topic_subtopic(x['Next Topic'], vectorizer_next_topic, model_next_topic, labels_next_topic, subtopics_next_topic_df)), axis=1)

            # Store data in session state
            st.session_state.input_df = input_df
            st.session_state.master_db = master_db
            st.session_state.master_db_path = master_db_path
            st.session_state.comments_file = comments_file

        # Retrieve data from session state
        input_df = st.session_state.input_df

        # Create separate DataFrames for each table
        comment_df = input_df[['Comment', 'Comment Sentiment', 'VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound']]
        improvement_df = input_df[['Improvement', 'Improvement Topic', 'Improvement Sub Topic']]
        next_topic_df = input_df[['Next Topic', 'Next Topic Category', 'Next Topic Sub Category']]

        # Create graphs and table summaries
        st.header(":blue[📊 Data Summaries and Visualizations]")

        # 1. Distribution of comment sentiment
        sentiment_counts = comment_df[comment_df['Comment'] != 'None']['Comment Sentiment'].value_counts()
        sentiment_percentages = sentiment_counts / sentiment_counts.sum() * 100
        sentiment_summary = pd.DataFrame({
            'Sentiment': sentiment_counts.index,
            'Count': sentiment_counts.values,
            'Percentage': [f"{x:.1f}%" for x in sentiment_percentages.values]
        })

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution of Comment Sentiment")
            fig = px.bar(sentiment_summary, x='Count', y='Sentiment', orientation='h', color='Sentiment',
                        color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red','none': 'black'})
            st.plotly_chart(fig)

        with col2:
            st.subheader("Comment Sentiment Summary")
            st.table(sentiment_summary)

        # 2. Distribution of Improvement Topic
        improvement_counts = improvement_df[(improvement_df['Improvement Topic'] != 'none') & (improvement_df['Improvement Sub Topic'] != 'none')]['Improvement Topic'].value_counts()
        improvement_percentages = improvement_counts / improvement_counts.sum() * 100
        improvement_summary = pd.DataFrame({
            'Topic': improvement_counts.index,
            'Count': improvement_counts.values,
            'Percentage': [f"{x:.1f}%" for x in improvement_percentages.values]
        })

        col1, col2 = st.columns(2)

        improvement_counts = improvement_counts.sort_values(ascending=True)

        with col1:
            st.subheader("Distribution of Improvement Topics")
            fig = px.bar(
                improvement_counts,
                x=improvement_counts.values,
                y=improvement_counts.index,
                labels={'x': 'Count', 'y': 'Improvement Topic'},
                orientation='h'
            )
            st.plotly_chart(fig)

        with col2:
            st.subheader("Improvement Topic Summary")
            st.table(improvement_summary)

        # 3. Distribution of Next Topic Category
        next_topic_counts = next_topic_df[(next_topic_df['Next Topic Category'] != 'none') & (next_topic_df['Next Topic Sub Category'] != 'none')]['Next Topic Category'].value_counts()
        next_topic_percentages = next_topic_counts / next_topic_counts.sum() * 100
        next_topic_summary = pd.DataFrame({
            'Topic': next_topic_counts.index,
            'Count': next_topic_counts.values,
            'Percentage': [f"{x:.1f}%" for x in next_topic_percentages.values]
        })

        col1, col2 = st.columns(2)

        next_topic_counts = next_topic_counts.sort_values(ascending=True)

        with col1:
            st.subheader("Distribution of Next Topic Categories")
            fig = px.bar(
                next_topic_counts,
                x=next_topic_counts.values,
                y=next_topic_counts.index,
                labels={'x': 'Count', 'y': 'Next Topic Category'},
                orientation='h'
            )
            st.plotly_chart(fig)

        with col2:
            st.subheader("Next Topic Category Summary")
            st.table(next_topic_summary)

        # Add filters for each table
        st.header(":blue[📋 Detailed Data]")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Comments and Sentiments")
            sentiment_filter = st.multiselect("Filter by Sentiment", options=comment_df['Comment Sentiment'].unique())
            if sentiment_filter:
                comment_df = comment_df[comment_df['Comment Sentiment'].isin(sentiment_filter)]
            st.dataframe(comment_df)

        with col2:
            st.subheader("Improvements and Topics")
            improvement_topic_filter = st.multiselect("Filter by Improvement Topic", options=improvement_df['Improvement Topic'].unique())
            improvement_subtopic_filter = st.multiselect("Filter by Improvement Sub Topic", options=improvement_df['Improvement Sub Topic'].unique())
            if improvement_topic_filter:
                improvement_df = improvement_df[improvement_df['Improvement Topic'].isin(improvement_topic_filter)]
            if improvement_subtopic_filter:
                improvement_df = improvement_df[improvement_df['Improvement Sub Topic'].isin(improvement_subtopic_filter)]
            st.dataframe(improvement_df)

        with col3:
            st.subheader("Next Topics and Categories")
            next_topic_category_filter = st.multiselect("Filter by Next Topic Category", options=next_topic_df['Next Topic Category'].unique())
            next_topic_subcategory_filter = st.multiselect("Filter by Next Topic Sub Category", options=next_topic_df['Next Topic Sub Category'].unique())
            if next_topic_category_filter:
                next_topic_df = next_topic_df[next_topic_df['Next Topic Category'].isin(next_topic_category_filter)]
            if next_topic_subcategory_filter:
                next_topic_df = next_topic_df[next_topic_df['Next Topic Sub Category'].isin(next_topic_subcategory_filter)]
            st.dataframe(next_topic_df)

        # Provide option for downloading the processed data
        st.download_button(
            label="Download Processed Data as CSV",
            data=input_df.to_csv(index=False).encode('utf-8'),
            file_name='processed_data.csv',
            mime='text/csv',
        )
        
    col = st.columns(1)
    with col[0]:
        if st.button("Back to Home Page"):
            st.session_state.page = "home"

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Navigation logic
if st.session_state.page == 'home':
    home()
elif st.session_state.page == 'CSAT, Lifecycle, Pulse, Townhall':
    csat_lifecycle_pulse_townhall()
elif st.session_state.page == 'Contact Center':
    contact_center()
elif st.session_state.page == 'Townhall':
    Townhall()
elif st.session_state.page == 'High Performance Processing':
    high_performance_processing()

