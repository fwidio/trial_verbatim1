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


# Set the page configuration to wide
st.set_page_config(layout="wide")

# Function to display the home page
def home():
    st.title("Welcome to the Python-Based Verbatim Tool")
    st.caption("Version updated 25 October 2024")
    st.caption("What's new? Contact Center Tool")
    st.caption("Note: Please double click the button")
    if st.button("CSAT, Lifecycle, Pulse, Townhall"):
        st.session_state.page = "CSAT, Lifecycle, Pulse, Townhall"
    if st.button("Contact Center"):
        st.session_state.page = "Contact Center"

# Function to display the CSAT, Lifecycle, Pulse, Townhall page
def csat_lifecycle_pulse_townhall():
    # Streamlit app layout
    st.title("Sentiment Analysis and Topic Prediction")
    
    # Download necessary NLTK data
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    # Initialize the sentiment intensity analyzer
    vader_analyzer = SentimentIntensityAnalyzer()

    # Language selection
    language = st.selectbox("Select Language", ["English", "Indonesia"])

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
        # Proceed with the rest of your code
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
            
            # Translate the comments if the selected language is Indonesian
            if language == "Indonesia":
                def translate_comment(comment):
                    try:
                        return translate(comment, 'en', 'id')
                    except Exception as e:
                        st.error(f"Translation error: {e}")
                        return comment

                comments_df['comment'] = comments_df['comment'].apply(lambda x: translate_comment(x) if pd.notna(x) else x)
                
            # Read custom lexicon from the master database
            custom_lexicon_df = pd.read_excel(master_db_path, sheet_name="custom lexicon")
            custom_lexicon = pd.Series(custom_lexicon_df['Sentiment Score'].values, index=custom_lexicon_df['Word']).to_dict()

            # Update VADER's lexicon with custom words
            vader_analyzer.lexicon.update(custom_lexicon)

            def clean_dataframe(df):
                # Remove rows that are empty or consist only of '-', '_', '0', 'none', or 'no'
                df.replace(['-', '_', '0', 'none', 'no','n/a','.',' ','`-','n.a.','..'], pd.NA, inplace=True)
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
        ##################################
                # Apply the categorization function to each row
                combined_df['Sentiment Category'] = combined_df.apply(categorize_sentiment, axis=1)
                
                # Add a selector for choosing between 'Topic' and 'Sub Topic'
                attribute_selector = st.selectbox("Select Attribute to Display", options=['Predicted Topic', 'Predicted Sub Topic'])

                # Visualization of top mentioned topics or subtopics
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
                    selected_subtopic = st.selectbox("Filter by Sub Topic", options=["All"] + filtered_subtopics)

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
                custom_stop_words = {'yet', 'quite','pmi','never','stage','took','needed'}  # Add custom stop words here
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
                    subtopic_filter = st.selectbox("Filter by  Sub Topic", options=["All", "undefined"] + labels)
                with col3:
                    topic_filter = st.selectbox("Filter by Topic", options=["All"] + subtopics_df['Category'].unique().tolist())

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
    st.header("Contact Center")
    st.write("Content for Contact Center page goes here.")

    # Function to clean the 'Subject' column
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

    # Streamlit app
    st.title("Topic & Sub Topic Finder")

    # Upload input file
    input_file = st.file_uploader("Upload Input File", type=["xlsx"])
    # Upload database file
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

        # Display the table dynamically
        st.write("Top 10 Topics with Counts and Percentages:")
        st.dataframe(top_10_table_data.style.set_properties(**{'text-align': 'center'}).set_table_styles(
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
