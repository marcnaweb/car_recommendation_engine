'''
    the following subroutines and classes are available in this module:     I desctiption of main subs
===================================================================================================
#####     1. import all relevant libraries,       #####                     I
#####     1.1 service subroutines               #####                       I
def where_am_I_running                                                      I
def get_base_directory_name                                                 I
def find_a_place_for_reviews_by_customers_clean                             I
def find_a_place_for_reviews_by_parkers                                     I
def find_a_place_for_trained_classification_models                          I
def find_a_place_for_summaries                                              I
def find_a_place_for_nltk_data_dir                                          I
def find_a_place_for_car_models_csv                                         I
def find_a_place_for_retreived_user_reviews_csv                             I
def create_folder_if_not_exists                                             I
def create_all_folders_if_not_exists                                        I
def flatten                                                                 I
def rm_multiple_spaces                                                      I
#####     2. train classifier using bag of words model, #####               I
def train_classifier_models_on_parkers_data <-------------------------------Ihead subrountine to train classifier
#####     2.1 read scraped reviews from Parkers        #####                I
def read_training_data_for_classification_models                            I
#####     2.2 LabelEncode target and store it          #####                I
def label_encode_target                                                     I
#####     2.3 subroutines to preprocess text           #####                I
class NLTKWordDownloader:                                                   I
    @classmethod                                                            I
    def download_wordnet                                                    I
def remove_punctuation                                                      I
def lowercase                                                               I
def remove_numbers                                                          I
def remove_stopwords                                                        I
def lemma                                                                   I
def all_preprocessing                                                       I
#####  2.4 classifier based on logistic regression      #####               I
#####               but with variable threshold         #####               I
class CustomClassifier                                                      I
    def __init__                                                            I
    def fit                                                                 I
    def predict                                                             I
    def transform                                                           I
#####  2.6  define & train pipelines                    #####               I
def define_and_train_pipelines                                              I
#####   3. split text in sentences and categorize them  #####               I
#####  3.1 subrotine to split text in sentences         #####               I
def categorize_text2<-------------------------------------------------------Ihead subrountine to categorize text
#####  3.2 service functions                            #####               I
 #        class that loads up all classification pipelines                  I
class CachedClassificationModels:                                           I
    def __init__                                                            I
    def retreive_classification_pipelines                                   I
    def _get_all_pipelines                                                  I
    def retreive_decode_dictionary                                          I
    def _get_decode_dictionary                                              I
def count_words                                                             I
def count_rows_exceeding_threshold                                          I
#####   4.    summarize concatenated user reviews text  #####               I
#####  4.1 categorize and summarize reviews             #####               I
def summarize_all_text<-----------------------------------------------------Ihead subrountine to summarize text
                                                                            I
#####  4.2 BART summarizer                              #####               I
def count_words_in_text_string                                              I
def select_words_from_text_string                                           I
class Summarizer:                                                           I
    def __init__                                                            I
    def summarize_text                                                      I
                                                                            I
#####   5.    load up user reviews of car models        #####               I
#            service functions                                              I
def retrieve_cust_rev1__cust_rev2<------------------------------------------Ihead subrountine to retrieve all reviews of models in model list with and without engine requirement relaxed
def read_from_HDD_cust_rev1__cust_rev2                                      I
def extract_first_word                                                      I
def extract_engine_capacity                                                 I
#####  5.1 make a dataframe of car models which need    #####               I
def read_unique_model_list<-------------------------------------------------Ihead subrountine to read model list
#####  5.2 loading up all customer reviews from HDD for #####               I
class CachedCustomerReviews:                                                I
    def __init__                                                            I
    def retreive_customer_reviews                                           I
    def _get_all_customer_reviews                                           I
def get_all_customer_reviews_for_model                                      I
def process_row_full_retrieval                                              I
def process_chunk_full_retreival                                            I
def retrieve_customer_reviews_for_models_in_database<-----------------------Ihead subrountine to retrieve all reviews of models in model list
#####  5.3 loading up all customer reviews from HDD for #####               I
def make_year_range                                                         I
def make_manufacturer_base_model_year_range                                 I
def return_reviews_of_cars_in_database_no_engine                            I
def process_row_full_retrieval_no_engine                                    I
def process_chunk_full_retreival_no_engine                                  I
def retrieve_customer_reviews_for_models_in_database_no_engine<-------------Ihead subrountine to retrieve all reviews of models in model list with engine requirement relaxed
#####   6.    categorize and summarize all              #####               I
def categorize_and_summarize_specific_user_reviews                          I
def process_row_cat_and_sum_cust_rev_all_entries                            I
def process_chunk_cat_and_sum_cust_rev_all_entries<-------------------------IProcess each row of dataframe to categorize and summarize reviews
#####   7.    test some of the methods              #####                   I
                                                                            I
def main<-------------------------------------------------------------------Itester
'''
#############################################################
#############################################################
#############################################################
#####                                                   #####
#####                                                   #####
#####           1. import all relevant libraries,       #####
#####           define  service subroutines             #####
#####                                                   #####
#####                                                   #####
#############################################################
#############################################################
#############################################################
import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import re
#scraping
import requests
from requests import get
from bs4 import BeautifulSoup
from bs4 import NavigableString #not used
from urllib.request import urlopen #not used
#generally useful libraries
import numpy as np
import pandas as pd
#saving and loading reviews
import json
#saving and loading models
import joblib
#bag - of - words modelling
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import string
#sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
#bart
import torch
from transformers import BartTokenizer, \
            BartForConditionalGeneration, \
            BartConfig,  \
            BartModel, \
            PegasusTokenizer, \
            PegasusForConditionalGeneration
from tensorflow import Tensor
#navigate to folders
import shutil
import time
import subprocess

# Define the number of CPU cores to use
from joblib import Parallel, delayed
#############################################################
#####                                                   #####
#####             1.1 service subroutines               #####
#####                                                   #####
#############################################################

#############################################################
#            subroutines to provide directory names
#############################################################
def where_am_I_running(debug=0):
    debug=1


    IN_COLAB = 'google.colab' in sys.modules
    IN_GCP = False
    LOCAL_COMPUTER = False
    if not IN_COLAB:
        #output = !hostname
        result = subprocess.run("hostname", shell=True, capture_output=True, text=True)


        output_str = result.stdout.strip()


        #if (debug):
        #    print (f"where_am_I_running: output = {output}")

        #output_str = " ".join(output)
        if  bool(re.search(r'-instance-20\d{2}', output_str)): IN_GCP = True
        if  bool(re.search('DESKTOP-', output_str)): LOCAL_COMPUTER = True
    if (debug):
        print (f"where_am_I_running response: IN_COLAB = {IN_COLAB},  IN_GCP = {IN_GCP},  LOCAL_COMPUTER = {LOCAL_COMPUTER}")

    if (IN_COLAB):
        return "Google colab"
    elif (IN_GCP):
        return "GCP"
    elif (LOCAL_COMPUTER):
        return "python library on local computer"
    #return "jupyter on local computer"
def get_base_directory_name():
    folder_name = "n/a"
    if (where_am_I_running() == "python library on local computer"):
        folder_name = os.path.dirname(os.path.realpath(__file__))
        folder_name = folder_name.replace(\
                    "/summarize_reviews",     \
                    "/data/")
    elif (where_am_I_running() == "jupyter on local computer"):
        folder_name = os.getcwd() +  "/data/"
    elif (where_am_I_running() == "Google colab"):
        folder_name = "/content/gdrive/MyDrive/Colab Notebooks/CarRecommendationEngine_classifySentences/data/"
    elif (where_am_I_running() == "GCP"):
        folder_name = "/home/romanz/app/data/"

    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name
def find_a_place_for_reviews_by_customers_clean():
    folder_name = get_base_directory_name() +  "/AllReviews_by_customers_clean/AllReviews_by_customers_clean.csv"
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name
def find_a_place_for_reviews_by_parkers ():
    folder_name = get_base_directory_name() +  "/reviews_from_parkers"
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name
def find_a_place_for_trained_classification_models ():
    folder_name = get_base_directory_name() +  "/trained_classification_models"
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name
def find_a_place_for_summaries ():
    folder_name = get_base_directory_name() +  "/summaries"
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name
def find_a_place_for_nltk_data_dir ():
    folder_name = get_base_directory_name() +  "/nltk_data_dir"
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name
def find_a_place_for_car_models_csv ():
    folder_name = get_base_directory_name() +  "/car_models_csv"
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name
def find_a_place_for_retreived_user_reviews_csv ():
    folder_name = get_base_directory_name() +  "/retreived_user_reviews/"
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
def create_all_folders_if_not_exists():
    create_folder_if_not_exists(find_a_place_for_reviews_by_customers_clean())
    create_folder_if_not_exists(find_a_place_for_reviews_by_parkers())
    create_folder_if_not_exists(find_a_place_for_trained_classification_models())
    create_folder_if_not_exists(find_a_place_for_summaries())
    create_folder_if_not_exists(find_a_place_for_nltk_data_dir())
    create_folder_if_not_exists(find_a_place_for_car_models_csv())
    create_folder_if_not_exists(find_a_place_for_retreived_user_reviews_csv())
#############################################################
#            reduce dimensions of a list
#############################################################
def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list
#############################################################
#            subrotine to remove multiple spaces from strings
#############################################################
def rm_multiple_spaces(text):
    # Split the text on spaces
    words = text.replace("\n", "NEWL").split()
    # Join the words back together with a single space between each word
    return ' '.join(words).replace("NEWL ","NEWL" ) .replace("NEWL","\n" )



#############################################################
#############################################################
#############################################################
#####                                                   #####
#####     2. train classifier using bag of words model, #####
#####       NLTK library on Parkers data (labelled)     #####
#####                                                   #####
#############################################################
#############################################################
#############################################################
def train_classifier_models_on_parkers_data(debug):
    #section 2.1
    if (debug): print ("calling read_training_data_for_classification_models")
    reviews_for_classification = \
        read_training_data_for_classification_models(debug-1)

    if (debug): print (f"reviews_for_classification =  {reviews_for_classification}")

    #section 2.2
    if (debug): print ("calling label_encode_target")
    reviews_for_classification = \
        label_encode_target (reviews_for_classification, debug-1)
    if (debug): print (f"reviews_for_classification =  {reviews_for_classification}")


    #section 2.6
    if (debug): print ("calling define_and_train_pipelines")
    define_and_train_pipelines(reviews_for_classification, debug-1)
    if (debug): print ("done define_and_train_pipelines")


#############################################################
#####                                                   #####
#####      2.1 read scraped reviews from Parkers        #####
#####                                                   #####
#############################################################
def read_training_data_for_classification_models(debug = 0):
    #############################################################
    #            read and preprocess labelled text
    #############################################################
    # Directory path
    directory = find_a_place_for_reviews_by_parkers()
    if (debug):
        print (f"directory = {directory}")
    # Get a list of files in the directory
    files = os.listdir(directory)
    all_review_list = []
    for current_file in files:
        with open(directory + "/" + current_file, 'r') as f:
            loaded_list = json.load(f)
            all_review_list.append(loaded_list)
    if (debug):
        print(f"all_review_list[-1] = {all_review_list[-1]}\n=================")
    all_review_list_flat = flatten(all_review_list)
    df_reviews = pd.DataFrame(all_review_list_flat)
    # Remove rows containing 'n/a' entries
    df_reviews = df_reviews[(df_reviews != 'n/a').all(axis=1)]
    #convert the dataframe into one column being review,
    # and another column being classification of that review
    df_only_reviews = \
        df_reviews.drop(columns=["manufacturer", "model"])\
        .reset_index(drop=True)
    reviews_for_classification = \
        df_only_reviews.melt(var_name='topic', value_name='review_text')
    if (debug):
        print(rm_multiple_spaces(f"reviews_for_classification = \n\
            {reviews_for_classification}\n\
                ================="))
    return reviews_for_classification

#############################################################
#####                                                   #####
#####      2.2 LabelEncode target and store it          #####
#####       into a column called  "topic_encoded"       #####
#####                                                   #####
#############################################################
def label_encode_target (reviews_for_classification, debug = 0):
    #encode labels
    le = LabelEncoder()
    le.fit(reviews_for_classification["topic"])
    reviews_for_classification["topic_encoded"] = \
        le.transform (reviews_for_classification["topic"])
    #############################################################
    #            create dictionary for reverse label transformation
    #############################################################
    # Apply filter to DataFrame - only keep zeroth row of each topic,
    # keep only columns "topic" and "topic_encoded"
    result = \
        reviews_for_classification.groupby('topic')\
            .first().reset_index().loc[:, ["topic","topic_encoded"]]
    # Make column 'topic_encoded' as the index
    result.set_index('topic_encoded', inplace=True)
    # convert df to dictionary
    decode_topic_dict = result.to_dict()['topic']
    if (debug):
        print(f"decode_topic_dict = \n{decode_topic_dict}\n=================")
    # Save dictionary to a JSON file
    dict_file = find_a_place_for_trained_classification_models() \
        + "/" +  "decode_topic_dict.json"
    if (debug):
        print(f"dict_file = \n{dict_file}\n=================")
    with open(dict_file , 'w') as f:
        json.dump(decode_topic_dict, f)
    return reviews_for_classification
#############################################################
#####                                                   #####
#####      2.3 subroutines to preprocess text           #####
#####                                                   #####
#############################################################
#create a class that handles the downloading of NLTK data and setting
# On first instantiation, download NLTK data
# On subsequent instantiations, the class will skip the download process.
class NLTKWordDownloader:
    _downloaded = False
    @classmethod
    def download_wordnet(cls):
        if not cls._downloaded:
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            cls._downloaded = True
def remove_punctuation(text, debug=0):
    if (debug): print (f"remove_punctuation text  ={text}")
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text
def lowercase (text, debug=0):
    if (debug): print (f"lowercase text  ={text}")
    lowercased = text.lower()
    return lowercased
def remove_numbers (text, debug=0):
    if (debug): print (f"remove_numbers text  ={text}")
    words_only = ''.join([i for i in text if not i.isdigit()])
    return words_only
def remove_stopwords (text, debug=0):
    if (debug): print (f"remove_stopwords text  ={text}")
    download_new_nltk_every_time = 0
    if (download_new_nltk_every_time):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    else:
        NLTKWordDownloader.download_wordnet()
    # From now on, NLTK data will be available
    # and NLTK_DATA environment variable will be set
    stop_words = set(stopwords.words('english'))
    tokenized = word_tokenize(text)
    without_stopwords = [word for word in tokenized if not word in stop_words]
    return without_stopwords
def lemma(text, debug=0):
    if (debug): print (f"lemma text  ={text}")
    lemmatizer = WordNetLemmatizer() # Instantiate lemmatizer
    lemmatized = [lemmatizer.lemmatize(word) for word in text] # Lemmatize
    lemmatized_string = " ".join(lemmatized)
    return lemmatized_string
def all_preprocessing(text, debug=0):
    debug=0
    preprocessed_texts = []
    # the subtourine accepts a numpy array of sentences
    # and returns a list. This allows it to be used in the pipeline
    for current_line in text:
        output = remove_punctuation(current_line, debug)
        output = lowercase(output)
        output = remove_numbers(output, debug)
        output = remove_stopwords(output, debug)
        output = lemma(output, debug)
        preprocessed_texts.append(output)
    return preprocessed_texts
#############################################################
#####                                                   #####
#####  2.4 classifier based on logistic regression      #####
#####               but with variable threshold         #####
#####                                                   #####
#############################################################
class CustomClassifier():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = LogisticRegression()
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        probabilities = self.model.predict_proba(X)
        max_probabilities = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        # Assign a separate class label for cases below threshold
        predictions[max_probabilities < self.threshold] = -1
        return predictions
    # This is a dummy method as FeatureUnion requires
    # all estimators to implement 'transform'
    def transform(self, X):
        return X
#############################################################
#####                                                   #####
#####  2.6  define & train pipelines                    #####
#####                                                   #####
#############################################################
def define_and_train_pipelines(reviews_for_classification, debug=0):
    # Create multiple instances of CustomClassifier with different thresholds
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, \
        0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    classifiers = \
        [CustomClassifier(threshold=threshold) for threshold in thresholds]
    # Create pipelines for each classifier
    pipelines = [Pipeline([
        ('preprocessor', FunctionTransformer(all_preprocessing)),  # Custom preprocessing step
        ('vectorizer', CountVectorizer()),
        ('classifier', classifier)
    ]) for classifier in classifiers]
    X_train, X_test, y_train, y_test = train_test_split(
            reviews_for_classification["review_text"].values,
            reviews_for_classification["topic_encoded"].values,
            test_size=0.05,
            random_state=42)
    print (f"X_train = {X_train}")
    # Train pipelines with differing thresholds
    for pipeline in pipelines:
        pipeline.fit(X_train, y_train)
    # Save the trained pipelines to a file
    #debug = 1
    for pipeline_no, pipeline in enumerate(pipelines):
        fname = find_a_place_for_trained_classification_models() + \
            "/" + f'trained_pipeline_{pipeline_no}.pkl'
        if (debug):
            print (f"fname = {fname}")
        joblib.dump(pipeline, fname)
#############################################################
#############################################################
#############################################################
#####                                                   #####
#####                                                   #####
#####   3. split text in sentences and categorize them  #####
#####                                                   #####
#####                                                   #####
#############################################################
#############################################################
#############################################################

#############################################################
#####                                                   #####
#####  3.1 subrotine to split text in sentences         #####
#####  and categorize it based on pre - trained         #####
#####   pipelines                                       #####
#####                                                   #####
#############################################################
def categorize_text2(user_reviews, TargetTextSize = 1000,  debug=0):
  #the following approach to categorize text is implemented:
  #(1) split text in categories with model #0 (with lowest precision);
  #(2) for text in each category, progressively increase precision until
  # TargetTextSize is reached
#############################################################
##                                                         ##
##    (1) split text in categories with                    ##
##    model #0 (with lowest precision)                     ##
##                                                         ##
#############################################################
#############################################################
#     text pre - processing
#############################################################
    sentences = sent_tokenize(user_reviews)
    # Convert list of sentences into a NumPy array
    sentences_df = pd.DataFrame(sentences)
    sentences_df.columns = ['sentence']
    if (debug):
        print (rm_multiple_spaces(\
            f"\n=================\n \
            sentences_df on the input = \n\
            {sentences_df }\n\
            =================\n"))
#############################################################
#     load up pipelines from HDD
#############################################################
    cached_class_pipelines_instance = CachedClassificationModels()
    pipelines = \
        cached_class_pipelines_instance.retreive_classification_pipelines()
#############################################################
#     rough sentence categorization with pipeline #0
#############################################################
    #apply pipeline to sentence and generate another column
    pipeline = pipelines[0]
    sentences_df['topic_pipeline0'] = pipeline.predict(sentences_df['sentence'])
    #make sure that each class is present in at least one row
    # of the dataframe, as otherwise summary won't work
    def add_row_to_df(df, values_in_row):
        new_row = pd.Series([values_in_row] * len(df.columns), index=df.columns)
        new_row["sentence"] =" "
        df = df.append(new_row, ignore_index=True)
        return df
    sentences_df = add_row_to_df(sentences_df, 0)
    sentences_df = add_row_to_df(sentences_df, 1)
    sentences_df = add_row_to_df(sentences_df, 2)
    sentences_df = add_row_to_df(sentences_df, 3)
    if (debug):
        print (rm_multiple_spaces(\
            f"\n=================\n \
            sentences_df  = \n\
            {sentences_df }\n\
            =================\n"))
#############################################################
# concatenate all sentences belonging to each topic
#############################################################
    concatenated_text_rough = \
        sentences_df\
        .groupby('topic_pipeline0')['sentence']\
        .apply(lambda x: ' '.join(x))
    concatenated_text_rough_df = pd.DataFrame(concatenated_text_rough)
    concatenated_text_rough_df = concatenated_text_rough_df.rename_axis('topic')
    concatenated_text_rough_df['word_count'] = \
        concatenated_text_rough_df['sentence'].apply(lambda x: count_words(x))
    if (debug):
        print (rm_multiple_spaces(\
            f"\n=================\n \
            concatenated_text_rough_df  = \n\
            {concatenated_text_rough_df }\n\
            =================\n"))
#############################################################
##                                                         ##
##     (2) for text in each category, progressively        ##
##     increase precision until                            ##
##      TargetTextSize is reached                          ##
##                                                         ##
#############################################################
    # Define a filter function
    def filter_function(text1, topic, TargetTextSize):
        # count number of words in the sentence - if it's below
        # TargetTextSize, return sentence unmodified
        if ( count_words(text1) <= TargetTextSize):
            return text1
        else: #
            #apply
            # progressively stricter classification models until
            # target number of words is reached
            # split text in sentences
            sentences1 = sent_tokenize(text1)
            sentences1_df = pd.DataFrame(sentences1)
            sentences1_df.columns = ['sentence']
            old_concat_text = text1
            num_pipelines = 13
            for pipeline_no in range (1, num_pipelines):
                pipeline = pipelines[pipeline_no]
                #predict topic with the new pipeline
                sentences1_df['topic1'] = \
                    pipeline.predict(sentences1_df['sentence'])
                # concatenate all sentences belonging to each topic
                concatenated_text1 = \
                    sentences1_df.groupby('topic1')['sentence']\
                    .apply(lambda x: ' '.join(x))
                concatenated_text1_df = pd.DataFrame(concatenated_text1)
                concatenated_text1_df = \
                    concatenated_text1_df.rename_axis('topic')
                #count words in topic of interest
                concatenated_text1_df['word_count'] = \
                    concatenated_text1_df['sentence']\
                    .apply(lambda x: count_words(x))
                concat_text = concatenated_text1_df.loc[int(topic), 'sentence']
                word_count = concatenated_text1_df.loc[int(topic), 'word_count']
                if ( word_count <= TargetTextSize):
                    return old_concat_text
                elif (pipeline_no == num_pipelines):
                    return concat_text
                else:
                    old_concat_text = concat_text
#############################################################
#     go through each topic of concatenated_text_rough.
#     If number of words is greater than target, apply
#     progressively stricter classification models until
#     target number of words is reached
#############################################################
    # Cycle through topics, applying filter_function to the set of filtered topics
    for topic in concatenated_text_rough_df.index:
        retrieved_string = \
            concatenated_text_rough_df.loc[int(topic), 'sentence']
        concatenated_text_rough_df.loc[int(topic), 'filtered_sentence'] =\
            filter_function(retrieved_string, topic, TargetTextSize)
    concatenated_text_rough_df['word_count_filtered'] = \
        concatenated_text_rough_df['filtered_sentence']\
        .apply(lambda x: count_words(x))
    if (debug):
        print (rm_multiple_spaces(\
            f"\n=================\n \
            concatenated_text_rough_df  = \n\
            {concatenated_text_rough_df }\n\
            =================\n"))
    # clean up the dataframe
    concatenated_text_rough_df.rename(\
        columns={'filtered_sentence': 'review_selected_text'}, inplace=True)
    concatenated_text_rough_df = \
        concatenated_text_rough_df.drop(columns=["sentence", "word_count", ])
    concatenated_text_rough_df['topic'] = concatenated_text_rough_df.index
    concatenated_text = concatenated_text_rough_df
#############################################################
#     Reverse label encoder
#############################################################
    decode_topic_dict = \
        cached_class_pipelines_instance.retreive_decode_dictionary()
    if (debug):
        print (f"decode_topic_dict = \n{decode_topic_dict}")
    #debug
    #decode_topic_dict = {
    # "0": 'engines',
    # "1": 'interior',
    # "2": 'mpg_running_costs',
    # "3": 'practicality'}
    # Convert column 'topic' from integer to string
    concatenated_text['topic'] = concatenated_text['topic'].astype(str)
    concatenated_text['topic_decoded'] = \
        concatenated_text.topic.map(decode_topic_dict)
    if (debug):
        print (rm_multiple_spaces(\
            f"\n=================\n \
            categorized concatenated_text with decoded label = \n\
            {concatenated_text }\n\
            =================\n"))
    return concatenated_text
#############################################################
#####                                                   #####
#####  3.2 service functions                            #####
#####                                                   #####
#############################################################
#############################################################
#        class that loads up all classification pipelines,
#        loads up decode dictionary from HDD on the
#        1st invocation; stores then in RAM and
#        returns them from RAM on further invocations
#############################################################
class CachedClassificationModels:
    _cached_pipelines = {}  # Class-level variable to hold the cached DataFrame
    _cached_dict = {}
    _debug = 0


    def __init__(self):
        pass


    def retreive_classification_pipelines(self):
        if not CachedClassificationModels._cached_pipelines:
            # Read pipelines from HDD on first instantiation
            CachedClassificationModels._cached_pipelines = self._get_all_pipelines(CachedClassificationModels._debug)
            if CachedClassificationModels._debug:
                print("DataFrame read from file")
        else:
            if CachedClassificationModels._debug:
                print("Using cached DataFrame")
        return CachedClassificationModels._cached_pipelines
    #############################################################
    #            loading up all classification pipelines from HDD
    #############################################################
    @staticmethod
    def _get_all_pipelines (debug=0):
        #load up all customer reviews for all models
        #load up pipelines from HDD
        pipelines = {}
        for pipeline_no in range (0,13):
            fname = find_a_place_for_trained_classification_models() + \
                "/" + f'trained_pipeline_{pipeline_no}.pkl'
            if (debug):
                print (f"loading pretrained classification model from file = \n{fname}")
            pipelines[pipeline_no] = joblib.load(fname)
        return pipelines
    def retreive_decode_dictionary(self):
        if not CachedClassificationModels._cached_dict:
            # Read decode dictionary from HDD on first instantiation
            CachedClassificationModels._cached_dict = self._get_decode_dictionary(CachedClassificationModels._debug)
            if CachedClassificationModels._debug:
                print("Decode dictionary read from file")
        else:
            if CachedClassificationModels._debug:
                print("Using cached decode dictionary")
        return CachedClassificationModels._cached_dict

    #############################################################
    #            loading up decode dictionary from HDD
    #############################################################
    @staticmethod
    def _get_decode_dictionary (debug=0):
        #load up all customer reviews for all models
        #load up pipelines from HDD
        dict_file = find_a_place_for_trained_classification_models() + \
            "/" +  "decode_topic_dict.json"
        with open(dict_file, 'r') as f:
            decode_topic_dict = json.load(f)
        #print (f"decode_topic_dict = {decode_topic_dict}")
        return decode_topic_dict


# Function to count words in a text
def count_words(text):
    return len(str(text).split())
# Function to count rows where number of words exceeds 100
def count_rows_exceeding_threshold(column):
    return sum(column.apply(count_words) > 100)
#############################################################
#############################################################
#############################################################
#####                                                   #####
#####   4.    summarize concatenated user reviews text  #####
#####                                                   #####
#############################################################
#############################################################
#############################################################

#############################################################
#####                                                   #####
#####  4.1 categorize and summarize reviews             #####
#####  Subroutine is written to take two reviews        #####
#####                                                   #####
#############################################################
#
#user_reviews_particular_engine (all reviews gathered for a model with specific engine)
#   |-> summary_review_of_this_model (not categorized)
#   |-> summary_review_particular_engine (engine category)
#
#
#user_reviews_all_engines (all reviews gathered for a model regardless of the engine)
#   |-> summary_engines
#   |-> summary_interior
#   |-> summary_mpg_running_costs
#   |-> summary_practicality
#

def summarize_all_text(user_reviews_particular_engine                   ="",\
                                    user_reviews_all_engines            ="",\
                                    summarize_review_of_this_model      =1, \
                                    summarize_review_particular_engine  =1, \
                                    summarize_engines                   =1, \
                                    summarize_interior                  =1, \
                                    summarize_mpg_running_costs         =1, \
                                    summarize_practicality              =1, \
                                    debug=0):
#############################################################
#     Summarize each categorized topic
#############################################################
    start_time = time.time()
    summarize_particular_engine = 1
    if (len(user_reviews_particular_engine) > 10):
        concatenated_text_particular_engine =  \
            categorize_text2(user_reviews_particular_engine, \
            TargetTextSize = 1000,  \
            debug=0)
    else:
        summarize_particular_engine = 0

    summarize_all_engines = 1
    if (len(user_reviews_all_engines) > 10):
        concatenated_text_all_engines       =  \
            categorize_text2(user_reviews_all_engines, \
            TargetTextSize = 1000,  \
            debug=0)
    else:
        summarize_all_engines = 0

    end_time = time.time(); time_categorize = end_time - start_time
    TargetTextSize = 1000
    summarizer = Summarizer()
    def summarize_topic(concatenated_text, topic, debug):
        if (debug):
            print (rm_multiple_spaces(
                f"concatenated_text['topic_decoded'] == '{topic}', \
                'review_selected_text'].iloc[0] = \n \
                {concatenated_text.loc[concatenated_text['topic_decoded'] == topic, 'review_selected_text'].iloc[0] }"))
        try:
            summary = summarizer.summarize_text(
                concatenated_text.loc[concatenated_text['topic_decoded'] == topic, \
                'review_selected_text'].iloc[0], \
                max_length=TargetTextSize, \
                limit_number_of_partial_summaries=1, \
                debug=debug)
        except:
            summary = "n/a"
        return summary


    summary_review_of_this_model = "n/a"
    summary_review_particular_engine = "n/a"
    summary_engines = "n/a"
    summary_interior = "n/a"
    summary_mpg_running_costs = "n/a"
    summary_practicality = "n/a"
    start_time = time.time()
    if (summarize_review_of_this_model and summarize_particular_engine):
        summary_review_of_this_model = \
            summarizer.summarize_text(user_reviews_particular_engine )
    if (summarize_review_particular_engine and summarize_particular_engine):
        summary_review_particular_engine = \
            summarize_topic(concatenated_text_particular_engine, "engines", debug)



    if (summarize_engines and summarize_all_engines):
        summary_engines = \
            summarize_topic(
                concatenated_text_all_engines, "engines", debug)
    if (summarize_interior and summarize_all_engines):
        summary_interior = \
            summarize_topic(
                concatenated_text_all_engines, "interior", debug)
    if (summarize_mpg_running_costs and summarize_all_engines):
        summary_mpg_running_costs = \
            summarize_topic(
                concatenated_text_all_engines, "mpg_running_costs", debug)
    if (summarize_practicality and summarize_all_engines):
        summary_practicality = \
            summarize_topic(
                concatenated_text_all_engines, "practicality", debug)


    end_time = time.time(); time_summarize = end_time - start_time
    current_model_summary = {}
    current_model_summary["summary_review_of_this_model"] =  summary_review_of_this_model
    current_model_summary["summary_review_particular_engine"] =  summary_review_particular_engine
    current_model_summary["summary_engines"] =  summary_engines
    current_model_summary["summary_interior"] =  summary_interior
    current_model_summary["summary_mpg_running_costs"] =  summary_mpg_running_costs
    current_model_summary["summary_practicality"] =  summary_practicality
    current_model_summary["time_categorize"] =  time_categorize
    current_model_summary["time_summarize"] =  time_summarize
    return current_model_summary
#############################################################
#####                                                   #####
#####  4.2 BART summarizer                              #####
#####                                                   #####
#############################################################
def count_words_in_text_string (text_in):
    return len(text_in.split())
def select_words_from_text_string (text_in, first_word=0, last_word=750):
    words = text_in.split()[first_word:last_word]
    selected_text = ' '.join(words)
    return selected_text
class Summarizer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")
        # Preload tokenizer and model
        self.tokenizer = \
            BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = \
            BartForConditionalGeneration.from_pretrained(
                'facebook/bart-large-cnn').to(self.device)
    def summarize_text (self, \
                        input_text, \
                        max_length=150, \
                        limit_number_of_partial_summaries=3, \
                        debug=0, \
                        model_type="use_pegasus"):
        if (count_words_in_text_string (input_text) <10):
            return "n/a"
        num_beams=4 #default
        #num_beams=1 #simpler
        # Ensure CUDA is available for acceleration
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA:", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")
        # Load the BART tokenizer and model
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
        # Encode the text input and convert to torch tensors
        inputs = self.tokenizer.encode("summarize: " + input_text,
                                       return_tensors="pt",
                                       max_length=512,
                                       truncation=True).to(self.device)
        # Generate summary with BART
        summary_ids = self.model.generate(inputs,
                                          max_length=max_length,
                                          min_length=40,
                                          length_penalty=2.0,
                                          num_beams=num_beams,
                                          early_stopping=True)
        # Decode and print the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        #print("Summary:", summary)
        return summary


#############################################################
#############################################################
#############################################################
#####                                                   #####
#####                                                   #####
#####   5.    load up user reviews of car models        #####
#####         which are available from car_models.csv   #####
#####         which are in the user review database,    #####
#####         make a summary and store it to the hdd    #####
#####                                                   #####
#####                                                   #####
#############################################################
#############################################################
#############################################################
def retrieve_cust_rev1__cust_rev2(df_unique_manufacturer_year_base_model_in=None,
                                  num_cores=8,
                                  save_to_hdd=1,
                                  read_inputs_from_hdd=1,
                                  debug=0):

    if df_unique_manufacturer_year_base_model_in is None:
        df_unique_manufacturer_year_base_model = read_unique_model_list()
    else:
        df_unique_manufacturer_year_base_model = df_unique_manufacturer_year_base_model_in

    if (read_inputs_from_hdd):
        fname1 = find_a_place_for_retreived_user_reviews_csv()+ \
            "/customer_reviews_for_models_in_database.csv"
        fname2 = find_a_place_for_retreived_user_reviews_csv()+ \
            "/customer_reviews_for_models_in_database_no_engine.csv"

        df_manufacturer_year_base_model_customer_review = pd.read_csv(fname1)
        df_retrieved_reviews_no_engine = pd.read_csv(fname2)

    else:
        df_manufacturer_year_base_model_customer_review = \
            retrieve_customer_reviews_for_models_in_database(
                df_unique_manufacturer_year_base_model,
                num_cores=num_cores,
                debug=debug)

        df_manufacturer_base_model_year_range =  make_manufacturer_base_model_year_range()

        df_retrieved_reviews_no_engine = \
            retrieve_customer_reviews_for_models_in_database_no_engine(\
                model_database=df_manufacturer_base_model_year_range,
                num_cores=num_cores,
                debug=debug)


    df_make_yr_mdl_cust_rev1__cust_rev2 = pd.concat(\
        [df_manufacturer_year_base_model_customer_review, \
            df_retrieved_reviews_no_engine], axis=1)

    df_make_yr_mdl_cust_rev1__cust_rev2 = pd.concat(\
        [df_unique_manufacturer_year_base_model, \
            df_make_yr_mdl_cust_rev1__cust_rev2], axis=1)

    if (debug):
        print (f"df_make_yr_mdl_cust_rev1__cust_rev2.tail() = \
            {df_make_yr_mdl_cust_rev1__cust_rev2.tail()}")

    df_make_yr_mdl_cust_rev1__cust_rev2.drop(df_make_yr_mdl_cust_rev1__cust_rev2.columns[[4, 6]], axis=1, inplace=True)


    df_make_yr_mdl_cust_rev1__cust_rev2.columns = ["car_manufacturer",\
        "car_model_year",\
        "base_model",\
        "engine",\
        "customer_review",\
        "customer_review_all_engines_wide_mdl_year",\
        ]
    if (save_to_hdd):
        fname = find_a_place_for_retreived_user_reviews_csv()+ \
            "/retreived_user_reviews_and_summarized_reviews.csv"
        df_make_yr_mdl_cust_rev1__cust_rev2.to_csv(fname)
    return  df_make_yr_mdl_cust_rev1__cust_rev2

def read_from_HDD_cust_rev1__cust_rev2():
    fname = find_a_place_for_retreived_user_reviews_csv()+ "/retreived_user_reviews_and_summarized_reviews.csv"
    df_make_yr_mdl_cust_rev1__cust_rev2 = pd.read_csv(fname)
    return  df_make_yr_mdl_cust_rev1__cust_rev2

#############################################################
#            service functions
#############################################################
#segment car model into base model and trim -  the model is
# assumed to be the first word, trim is everything that follows
def extract_first_word(input_string):
    # Find the index of the first space
    first_space_index = input_string.find(" ")
    # Extract the substring up to the first space
    first_word  = input_string[:first_space_index]
    # Extract the substring up to the first space
    all_except_first_word  = input_string[first_space_index:]
    return first_word, all_except_first_word.strip()
import re
def extract_engine_capacity (input_string):
    # Regular expression to match numbers with two digits and a decimal point
    pattern = r'\b\d{1}\.\d{1}\b'
    # Extract numbers matching the pattern
    try:
        matches = re.findall(pattern, input_string)[0]
    except:
        matches = "n/a"
    return matches.strip()
#############################################################
#####                                                   #####
#####  5.1 make a dataframe of car models which need    #####
##### reviews, produce for every model the fields       #####
##### "car_manufacturer", "car_model_year",             #####
##### "base_model", "engine"                            #####
#####                                                   #####
#############################################################
def read_unique_model_list():
    f_car_models = find_a_place_for_car_models_csv () + "/car_models.csv"
    df_car_models = pd.read_csv(f_car_models)
    df_car_models[["base_model", "trim"]] = \
        df_car_models.apply(lambda row: extract_first_word(row['car_model']),
                            axis=1, result_type='expand')
    df_car_models["engine"] = \
        df_car_models.apply(lambda row: extract_engine_capacity(row['trim']),
                            axis=1, result_type='expand')
    #we want to make a summary of unique manufacturer + model year + base model
    unique_manufacturer_year_base_model = \
        ~df_car_models.duplicated(subset=[\
        "car_manufacturer", \
        "car_model_year",\
        "base_model",\
        "engine"], keep=False)
    # Keep only the rows with unique values
    df_unique_manufacturer_year_base_model = \
        df_car_models[unique_manufacturer_year_base_model]
    #Keep only the 3 specified columns
    df_unique_manufacturer_year_base_model = \
        df_unique_manufacturer_year_base_model[[\
        "car_manufacturer",\
        "car_model_year",\
        "base_model",\
        "engine"]]
    # Renumber the rows sequentially starting from 0
    df_unique_manufacturer_year_base_model = \
        df_unique_manufacturer_year_base_model.reset_index(drop=True)
    return df_unique_manufacturer_year_base_model
#############################################################
#####                                                   #####
#####  5.2 loading up all customer reviews from HDD for #####
#####      models in database (keeping the correct      #####
#####      manufacturer, model, year and engine)        #####
#####                                                   #####
#############################################################
#create a class that handles reading customer reviews from HDD
# On first instantiation, reading customer reviews from HDD
# On subsequent instantiations, the class will skip the read from HDD
# and return cached dataframe from RAM

class CachedCustomerReviews:
    _cached_df = None  # Class-level variable to hold the cached DataFrame
    def __init__(self):
        pass
    def retreive_customer_reviews (self):
        if CachedCustomerReviews._cached_df is None:
            # Read DataFrame from file on first instantiation
            CachedCustomerReviews._cached_df = self._get_all_customer_reviews()
            print("DataFrame read from file")
        else:
            print("Using cached DataFrame")
        return CachedCustomerReviews._cached_df
    #############################################################
    #            loading up all customer reviews from HDD
    #############################################################
    def _get_all_customer_reviews (debug=0):
        #load up all customer reviews for all models
        AllReviews_by_customers_clean_fname  =  \
            find_a_place_for_reviews_by_customers_clean()
        df_all_car_reviews_by_customers = \
            pd.read_csv(AllReviews_by_customers_clean_fname)
        df_all_car_reviews_by_customers.columns = ['key', \
            'manufacturer', \
            'old_key', \
            'review_date',\
            'reviewer_name', \
            'model', \
            'review_title',\
            'review_body', \
            'stars']
        df_all_car_reviews_by_customers.drop(columns=['key', \
            'old_key', 'reviewer_name'], inplace=True)
        df_all_car_reviews_by_customers['manufacturer'] = \
            df_all_car_reviews_by_customers.manufacturer.apply(lowercase)
        df_all_car_reviews_by_customers['model'] = \
            df_all_car_reviews_by_customers.model.apply(lowercase)
        #if (debug):
        #    print (f"df_all_car_reviews_by_customers = \n{df_all_car_reviews_by_customers}\n===============")
        return df_all_car_reviews_by_customers


#############################################################
# loading up customer reviews from HDD for a particular model
#############################################################
def get_all_customer_reviews_for_model(manufacturer,
                                       model,
                                       model_year="",
                                       engine="", debug=0):

    car_manufacturer_lookup = str(manufacturer).lower()
    car_model_year_lookup   = str(model_year).lower()
    base_model_lookup       = str(model).lower()
    engine_lookup           = str(engine).lower()
    if (model_year == "n/a"): car_model_year_lookup = ""
    if (engine == "n/a"): engine_lookup = ""

    cached_customer_reviews_instance = CachedCustomerReviews()
    df_all_car_reviews_by_customers = \
        cached_customer_reviews_instance.retreive_customer_reviews()
    #df_all_car_reviews_by_customers = get_all_customer_reviews (debug)
    filtered_df = df_all_car_reviews_by_customers[\
        (df_all_car_reviews_by_customers['model'].str.contains(base_model_lookup)) & \
        (df_all_car_reviews_by_customers['model'].str.contains(car_model_year_lookup)) & \
        (df_all_car_reviews_by_customers['model'].str.contains(engine_lookup)) & \
        (df_all_car_reviews_by_customers['manufacturer'].str.contains(car_manufacturer_lookup))]
    concatenated_string = filtered_df['review_body'].str.cat(sep='. ')

    if (len(concatenated_string) < 30):
        concatenated_string = "n/a"

    return concatenated_string



# Custom function to apply to each row
def process_row_full_retrieval(row):
    return get_all_customer_reviews_for_model \
            (row['car_manufacturer'], \
            row['base_model'], \
            row['car_model_year'], \
            row['engine'], \
            )
# Define the function to process a chunk of data
def process_chunk_full_retreival(chunk):
    # Process each row in the chunk using a lambda function
    processed_rows = \
        chunk.apply(lambda row: process_row_full_retrieval(row), axis=1)
    return processed_rows

def retrieve_customer_reviews_for_models_in_database(\
        model_database,\
        num_cores=8,\
        save_to_hdd=1, \
        debug=0):
    # Evenly divide DataFrame rows among cores
    chunk_size = len(model_database) // num_cores
    chunks = [
        model_database[i:i+chunk_size] for i in range(
            0, len(model_database), chunk_size)]
    # Process each chunk in parallel
    start_time = time.time()
    processed_chunks = Parallel(                                        \
        n_jobs=num_cores                                                \
        )(                                                              \
        delayed(process_chunk_full_retreival)(chunk) for chunk in chunks\
        )
    end_time = time.time()
    # Concatenate the processed chunks back into a single DataFrame
    processed_df = pd.concat(processed_chunks)
    # Convert results to DataFrame if needed
    if (debug):
        print(f"processed_df = \n{processed_df}\n===============\n")
    elapsed_time = end_time - start_time
    if (debug):
        print(f"num_cores = {num_cores}, Elapsed time: {elapsed_time} seconds")

    # Save the processed_df to a file
    if (save_to_hdd):
        fname = find_a_place_for_retreived_user_reviews_csv()+ \
            "/customer_reviews_for_models_in_database.csv"
        processed_df.to_csv(fname)

    return processed_df
#############################################################
#####                                                   #####
#####  5.3 loading up all customer reviews from HDD for #####
#####      models in database (keeping the correct      #####
#####      manufacturer, model,                         #####
#####      but any engine and model year                #####
#####      within 2 years of target)                    #####
#####                                                   #####
#############################################################
def make_year_range (car_model_year):
    return int(car_model_year) - 2, int(car_model_year)  + 2

def make_manufacturer_base_model_year_range (input_df=None):

    if input_df is None:
        df_manufacturer_base_model_year_range = read_unique_model_list()
    else:
        df_manufacturer_base_model_year_range = input_df

    df_manufacturer_base_model_year_range[["model_year_min", "model_year_max"]] = \
        df_manufacturer_base_model_year_range.apply(
            lambda row: make_year_range(
                row['car_model_year']),
            axis=1, result_type='expand')
    df_manufacturer_base_model_year_range.reset_index(drop=True, inplace=True)
    df_manufacturer_base_model_year_range = df_manufacturer_base_model_year_range.drop_duplicates()
    return df_manufacturer_base_model_year_range


#return review of model from database
def return_reviews_of_cars_in_database_no_engine (car_manufacturer, \
    car_model_year_min, \
    car_model_year_max, \
    base_model, \
    engine):
    all_user_reviews = ""
    for car_model_year in (car_model_year_min, car_model_year_max + 1):
        car_manufacturer_lookup = str(car_manufacturer).lower()
        car_model_year_lookup   = str(car_model_year).lower()
        base_model_lookup       = str(base_model).lower()
        engine_lookup           = ""
        debug = 0
        if (debug):
          print ( rm_multiple_spaces(f"\
            car_manufacturer_lookup = {car_manufacturer_lookup} \n\
            car_model_year_lookup   = {car_model_year_lookup  } \n\
            base_model_lookup       = {base_model_lookup      } \n\
            engine_lookup           = {engine_lookup          } \n\
                "))
        user_review = get_all_customer_reviews_for_model(\
            car_manufacturer_lookup, \
            car_model_year_lookup, \
            base_model_lookup, \
            engine_lookup)
        all_user_reviews = all_user_reviews + user_review
    if (len(user_review) < 30):
        return "n/a"
    else:
        return all_user_reviews


# Custom function to apply to each row
def process_row_full_retrieval_no_engine(row):
    # Perform your processing here
    return return_reviews_of_cars_in_database_no_engine \
            (row['car_manufacturer'], \
            row['model_year_min'], \
            row['model_year_max'], \
            row['base_model'], \
            "",\
            )
# Define the function to process a chunk of data
def process_chunk_full_retreival_no_engine(chunk):
    # Process each row in the chunk using a lambda function
    processed_rows = chunk.apply(lambda row: process_row_full_retrieval_no_engine(row), axis=1)
    return processed_rows



def retrieve_customer_reviews_for_models_in_database_no_engine(\
        model_database, \
        num_cores=8, \
        save_to_hdd=1, \
        debug=0):
    chunk_size = len(model_database) // num_cores  # Evenly divide DataFrame rows among cores
    chunks = [
        model_database[i:i+chunk_size] for i in range(
            0, len(model_database), chunk_size)]

    # Process each chunk in parallel
    start_time = time.time()
    processed_chunks = Parallel           \
                        (n_jobs=num_cores)\
                        (delayed          \
                            (process_chunk_full_retreival_no_engine)(chunk) \
                        for chunk in chunks)
    end_time = time.time()
    # Concatenate the processed chunks back into a single DataFrame
    processed_df = pd.concat(processed_chunks)

    if (save_to_hdd):
        fname = find_a_place_for_retreived_user_reviews_csv()+ \
            "/customer_reviews_for_models_in_database_no_engine.csv"
        processed_df.to_csv(fname)

    if (debug):
        print(f"processed_df = \n{processed_df}\n===============\n")
    elapsed_time = end_time - start_time
    if (debug):
        print(f"num_cores = {num_cores}, Elapsed time: {elapsed_time} seconds")
    return processed_df

#############################################################
#############################################################
#############################################################
#####                                                   #####
#####                                                   #####
#####   6.    categorize and summarize all              #####
#####         loaded user reviews                       #####
#####                                                   #####
#############################################################
#############################################################
#############################################################
def categorize_and_summarize_specific_user_reviews (row):
    debug = 1
    if isinstance(row, pd.DataFrame):
        car_manufacturer  = str(row.iloc[0, 0])
        car_model_year    = str(row.iloc[0, 1])
        base_model        = str(row.iloc[0, 2])
        engine            = str(row.iloc[0, 3])
        user_reviews_particular_engine = str(row.iloc[0, 4])
        user_reviews_all_engines = str(row.iloc[0, 5])
    elif isinstance(row, dict):
        car_manufacturer = row['car_manufacturer']
        car_model_year = row['car_model_year']
        base_model = row['base_model']
        engine = row['engine']
        user_reviews_particular_engine = row['customer_review']
        user_reviews_all_engines = row['customer_review_all_engines_wide_mdl_year']
    else:
        raise ValueError("Input must be a DataFrame or a dictionary")
    if (debug): print (rm_multiple_spaces(f"\
            car_manufacturer  = {car_manufacturer}\n\
            car_model_year    = {car_model_year  }\n\
            base_model        = {base_model      }\n\
            engine            = {engine          }\n\
            user_reviews_particular_engine = {user_reviews_particular_engine}\n\
            user_reviews_all_engines = {user_reviews_all_engines}\n   \
            "))
    summarized_reviews = \
        summarize_all_text(\
            user_reviews_particular_engine = user_reviews_particular_engine,\
            user_reviews_all_engines = user_reviews_all_engines,\
            debug=debug)
    if (debug): print (rm_multiple_spaces(f"\
            summarized_reviews['time_categorize']  =   \n\
                {summarized_reviews['time_categorize']}\n\
            summarized_reviews['time_summarize']  =    \n\
                {summarized_reviews['time_summarize']}\n\
            "))
    return [summarized_reviews["summary_review_of_this_model"]     ,\
            summarized_reviews["summary_review_particular_engine"] ,\
            summarized_reviews["summary_engines"]                  ,\
            summarized_reviews["summary_interior"]                 ,\
            summarized_reviews["summary_mpg_running_costs"]        ,\
            summarized_reviews["summary_practicality"]             ,\
            summarized_reviews["time_categorize"]                  ,\
            summarized_reviews["time_summarize"]                   ,\
            ]
# Custom function to apply to each row
def process_row_cat_and_sum_cust_rev_all_entries(row):
    # Perform your processing here
    print (f"type(row) = {type(row)}")
    review_model  = categorize_and_summarize_specific_user_reviews(row.to_dict())
    row['review_model']            = review_model[0]
    row['review_part_engine']      = review_model[1]
    row['review_summary_engines']  = review_model[2]
    row['review_summary_interior'] = review_model[3]
    row['review_summary_costs']    = review_model[4]
    row['review_summary_pract']    = review_model[5]
    return row
def process_chunk_cat_and_sum_cust_rev_all_entries(chunk, save_to_hdd=1):
    chunk = chunk.fillna("n/a")

    # Process each row in the chunk using the process_row_cat_and_sum_cust_rev_all_entries function directly
    processed_rows = chunk.apply(process_row_cat_and_sum_cust_rev_all_entries, axis=1)
    # Create a backup of the DataFrame every 30 rows
    if chunk.index[-1] % 30 == 0:
        backup_df = processed_rows.copy()
        backup_df.to_csv(f'backup_{chunk.index[-1]}.csv', index=False)  # Save the backup to a CSV file


    if (save_to_hdd):
        fname = find_a_place_for_retreived_user_reviews_csv()+ \
            "/categorized_and_summarized_customer_reviews_all_entries.csv"
        processed_rows.to_csv(fname)

    return processed_rows
#
#def categorize_and_summarize_customer_reviews_all_entries(model_database, \
#    num_cores=8, \
#    debug=0):
#
#    model_database = model_database.fillna("n/a")
#
#    chunk_size = len(model_database) // num_cores  # Evenly divide DataFrame rows among cores
#    chunks = [model_database[i:i + chunk_size] for i in range(0, len(model_database), chunk_size)]
#
#    # Process each chunk in parallel using multiprocessing.Pool
#    start_time = time.time()
#    with mp.get_context("spawn").Pool(num_cores) as pool:
#        processed_chunks = pool.map(process_chunk_cat_and_sum_cust_rev_all_entries, chunks)
#    end_time = time.time()
#
#    # Concatenate the processed chunks back into a single DataFrame
#    processed_df = pd.concat(processed_chunks)
#
#    if debug:
#        print(f"processed_df = \n{processed_df}\n===============\n")
#
#    elapsed_time = end_time - start_time
#    if debug:
#        print(f"num_cores = {num_cores}, Elapsed time: {elapsed_time} seconds")
#
#    return processed_df
#
#############################################################
#############################################################
#############################################################
#####                                                   #####
#####                                                   #####
#####   7.    test some of the methods                  #####
#####         loaded user reviews                       #####
#####                                                   #####
#############################################################
#############################################################
#############################################################
def main( ):
    create_all_folders_if_not_exists()

    '''tester'''
    what_to_test = "process_chunk_cat_and_sum_cust_rev_all_entries"

    if ("train_classifier_models" in what_to_test):
        print ("testing train_classifier_models_on_parkers_data")
        train_classifier_models_on_parkers_data(debug = 1)
        print ("testing train_classifier_models_on_parkers_data done")


    if ("categorize_text2" in what_to_test):
        print ("testing categorize_text2")


        cached_customer_reviews_instance = CachedCustomerReviews()
        BBB = cached_customer_reviews_instance.retreive_customer_reviews()
        CCC = ""
        for i in range(0,40):
            CCC = CCC + BBB.loc[i, "review_body"]


        print (f"CCC = {CCC}")



        concatenated_text = categorize_text2(user_reviews = CCC)
        print (f"concatenated_text = {concatenated_text}")
        print ("testing categorize_text2 done")

    if ("summarize_all_text" in what_to_test):
        print ("testing summarize_all_text")

        cached_customer_reviews_instance = CachedCustomerReviews()
        BBB = cached_customer_reviews_instance.retreive_customer_reviews()
        CCC = ""
        for i in range(0,40):
            CCC = CCC + BBB.loc[i, "review_body"]

        cached_customer_reviews_instance = CachedCustomerReviews()
        BBB = cached_customer_reviews_instance.retreive_customer_reviews()
        DDD = ""
        for i in range(40,80):
            DDD = DDD + BBB.loc[i, "review_body"]



        print (f"=======================\nCCC = \n{CCC}\n=======================\n")
        print (f"=======================\nDDD = \n{DDD}\n=======================\n")


        summary = summarize_all_text(user_reviews_particular_engine = CCC,
                                     user_reviews_all_engines = DDD)
        print (f"summary = {summary}")

        print ("testing summarize_all_text done")

    if ("read_unique_model_list" in what_to_test):

        print ("testing read_unique_model_list")

        df_unique_manufacturer_year_base_model = read_unique_model_list()
        print (f"df_unique_manufacturer_year_base_model = {df_unique_manufacturer_year_base_model}")
        print ("testing read_unique_model_list done")



    if ("retrieve_customer_reviews_for_models_in_database1" in what_to_test):
        print ("testing retrieve_customer_reviews_for_models_in_database")
        df_unique_manufacturer_year_base_model = read_unique_model_list()

        processed_df = retrieve_customer_reviews_for_models_in_database(df_unique_manufacturer_year_base_model)
        print (f"processed_df = {processed_df}")
        print ("testing retrieve_customer_reviews_for_models_in_database done")

    if ("retrieve_customer_reviews_for_models_in_database_no_engine" in what_to_test):
        print ("testing retrieve_customer_reviews_for_models_in_database_no_engine")

        df_manufacturer_base_model_year_range =  make_manufacturer_base_model_year_range()
        print (f"df_manufacturer_base_model_year_range = {df_manufacturer_base_model_year_range}")

        processed_df = retrieve_customer_reviews_for_models_in_database_no_engine(df_manufacturer_base_model_year_range)
        print (f"processed_df = {processed_df}")
        print ("testing retrieve_customer_reviews_for_models_in_database done")



    if ("retrieve_cust_rev1__cust_rev2" in what_to_test):
        print ("testing retreive_cust_rev1__cust_rev2")

        df_make_yr_mdl_cust_rev1__cust_rev2 = retrieve_cust_rev1__cust_rev2()

        print (f"df_make_yr_mdl_cust_rev1__cust_rev2.head(400).tail() = \
            {df_make_yr_mdl_cust_rev1__cust_rev2.head(400).tail()}")
        print ("testing retrieve_cust_rev1__cust_rev2 done")



    if ("process_chunk_cat_and_sum_cust_rev_all_entries" in what_to_test):
        df_make_yr_mdl_cust_rev1__cust_rev2 = retrieve_cust_rev1__cust_rev2()

        test_df = df_make_yr_mdl_cust_rev1__cust_rev2.head(400).tail(4)


        print ("testing process_chunk_cat_and_sum_cust_rev_all_entries")

        print (f"test_df = {test_df}")


        processed_df = process_chunk_cat_and_sum_cust_rev_all_entries(test_df)

        print (f"processed_df = {processed_df}")
        print ("testing process_chunk_cat_and_sum_cust_rev_all_entries done")




if __name__ == '__main__':
    main()
