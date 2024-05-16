import pandas as pd
from bs4 import BeautifulSoup
import requests
import os
import re
import string
import nltk

# Function to extract words from a file
def extract_words_from_file(file_path):
    with open(file_path, 'r') as file:
        # Split words by space and filter out single-letter words except 'a'
        words = file.read().split()
        return set([word for word in words if len(word) > 1 or word.lower() == 'a'])

# Function to extract words from all files in a folder
def extract_words_from_folder(folder_path):
    stop_words = set()  # Using set() to store unique words
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            words = extract_words_from_file(file_path)
            stop_words.update(words)  # Merge sets and remove duplicates
    return stop_words

# Function to create a dictionary of words from files in a folder
def create_word_dictionary(folder_path):
    word_dictionary = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        category = file_name.split('.')[0]  # Extract category from file name
        with open(file_path, 'r') as file:
            # Read words and create a set
            words = set(word.strip() for word in file.readlines())
        word_dictionary[category] = words  # Add category and corresponding set of words to dictionary
    return word_dictionary

# Function to count occurrences of words from dictionaries in a paragraph
def count_word_occurrences(paragraph, word_dictionary):
    paragraph = paragraph.lower()  # Convert paragraph to lowercase for case-insensitive matching
    positive_score = 0  # Initialize positive score
    negative_score = 0  # Initialize negative score
    for category, words in word_dictionary.items():
        for word in words:
            if category == "positive-words":
                positive_score += paragraph.count(word)  # Count occurrences of positive words
            elif category == "negative-words":
                negative_score += paragraph.count(word)  # Count occurrences of negative words
    return positive_score, negative_score

# Function to count complex words in a paragraph
def count_complex_words(paragraph):
    complex_words = []
    words = paragraph.split()  # Tokenize the paragraph into words
    for word in words:
        if len(word) > 2:  # Check if word is longer than 2 characters
            syllable_count = count_syllables(word)
            if syllable_count > 2:  # Check if word has more than 2 syllables
                complex_words.append(word)
    return len(complex_words)

# Function to count syllables in a word
def count_syllables(word):
    vowels = 'aeiouy'
    count = 0
    prev_char = ''
    for char in word:
        char = char.lower()
        if char in vowels and prev_char not in vowels:
            count += 1
        prev_char = char

    # Adjust syllable count for certain endings
    if word.endswith(('es', 'ed')):
        count -= 1

    # Ensure at least one syllable for short words
    count = max(count, 1)

    return count

# Function to calculate average word length in a paragraph
def calculate_average_word_length(paragraph):
    words = paragraph.split()  # Tokenize the paragraph into words
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    return total_characters / total_words

# Function to count average number of words per sentence in a text
def count_average_words_per_sentence(text):
    # Split the text into sentences using regular expressions
    sentences = re.split(r'[.!?]', text)
    
    # Remove empty strings from the list (caused by consecutive punctuation marks)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # Calculate the total number of words
    total_words = sum(len(sentence.split()) for sentence in sentences)
    
    # Calculate the total number of sentences
    total_sentences = len(sentences)
    
    # Calculate the average number of words per sentence
    average_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
    
    return average_words_per_sentence

# Function to count average syllables per word in a paragraph
def count_average_syllables_in_word(paragraph):
    total_syllables = 0
    total_words = 0
    vowels = 'aeiouy'

    words = paragraph.split()  # Split the paragraph into words
    for word in words:
        word = word.lower()
        word_syllables = 0
        prev_char = ''
        for char in word:
            char = char.lower()
            if char in vowels and prev_char not in vowels:
                word_syllables += 1
            prev_char = char

        # Adjust syllable count for certain endings
        if word.endswith(('es', 'ed')):
            word_syllables -= 1

        # Ensure at least one syllable for short words
        word_syllables = max(word_syllables, 1)

        total_syllables += word_syllables
        total_words += 1

    # Calculate average syllables per word
    average_syllables_per_word = total_syllables / total_words if total_words > 0 else 0
    return average_syllables_per_word

# Function to calculate average sentence length in a text
def average_sentence_length(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Count the number of words in each sentence
    word_counts = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    
    # Calculate the average sentence length
    if len(word_counts) > 0:
        average_length = sum(word_counts) / len(word_counts)
    else:
        average_length = 0
    
    return average_length

# Function to count personal pronouns in a text
def count_personal_pronouns(text):
    # Define the regex pattern to match personal pronouns
    pattern = r'\b(I|we|my|ours|us)\b'
    
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    
    # Count the total number of matches
    count = len(matches)
    
    return count

# Function to export data to an Excel file
def export_to_excel(filename, data):
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)
    
    # Export the DataFrame to an Excel file
    df.to_excel(filename, index=False)

# Define paths to stop words folder and master dictionary folder
folder_path = './StopWords'
folder_path2 = './MasterDictionary'

# Extract stop words and create word dictionary
stop_words = extract_words_from_folder(folder_path)
word_dictionary = create_word_dictionary(folder_path2)

# Read URLs from input Excel file
df = pd.read_excel('.\Input.xlsx')
URL = []
data = []

# Iterate through each URL and extract data
for index, row in df.iterrows():
    URL.append(row['URL'])

count = 0
for url in URL:
    try:
        # Get the response from the URL
        response = requests.get(url)
        
        # Check if the response is successful
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the title of the article
        title = soup.find('h1').get_text()

        # Check if the main paragraph class is present
        if soup.find(class_='td-post-content tagdiv-type') is None:
            para_class = 'td_block_wrap tdb_single_content tdi_130 td-pb-border-top td_block_template_1 td-post-content tagdiv-type'
        else:
            para_class = 'td-post-content tagdiv-type'

        # Find the paragraph(s) of the article
        para = soup.find(class_=para_class).get_text()

        # Further processing if successful
        text = para.split('Blackcoffer Insights')[0].strip()
        text = text.lower()
        text = ''.join(char for char in text if char not in string.punctuation)
        for word in stop_words:
            text = re.sub(r'\b' + re.escape(word.lower()) + r'\b', '', text)  # Remove word only if it's a whole word
            text.strip()

        # Calculate all metrics
        positive_score, negative_score = count_word_occurrences(text, word_dictionary)
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score) / ((len(text.split())) + 0.000001)
        complex_word_count = count_complex_words(text)
        word_count = len(text.split())
        avg_sentence_length = round(average_sentence_length(para))
        complex_word_percent = (complex_word_count / word_count) * 100
        fog_index = 0.4 * (avg_sentence_length + complex_word_percent)
        syllable_per_word = round(count_average_syllables_in_word(text))
        personal_pronoun_count = count_personal_pronouns(para)
        word_length = round(calculate_average_word_length(text))
        avg_num_of_words_per_sentence = count_average_words_per_sentence(para)

        # Append the calculated values to the data list
        data.append({
            "URL": url,
            "Positive Score": positive_score,
            "Negative Score": negative_score,
            "Polarity Score": polarity_score,
            "Subjectivity Score": subjectivity_score,
            "Complex Word Count": complex_word_count,
            "Word Count": word_count,
            "Average Sentence Length": avg_sentence_length,
            "Complex Word Percentage": complex_word_percent,
            "Fog Index": fog_index,
            "Syllables Per Word": syllable_per_word,
            "Personal Pronoun Count": personal_pronoun_count,
            "Word Length": word_length,
            "Average Number of Words Per Sentence": avg_num_of_words_per_sentence
        })

    except requests.exceptions.RequestException as e:
        print("Skipping page:", url, "- Error:", e)
    count += 1
    print(count)

# Export the data list to an Excel file
export_to_excel("output.xlsx", data)


# response = requests.get(URL[0])
# print(response.raise_for_status())
# soup = BeautifulSoup(response.content,'html.parser')
# title = soup.find('h1').get_text()
# para = soup.find(class_='td-post-content tagdiv-type').get_text()
# print(para)
# text = para.split('Blackcoffer Insights')[0].strip()
# text = text.lower()
# text = ''.join(char for char in text if char not in string.punctuation)
# for word in stop_words:
#     text = re.sub(r'\b' + re.escape(word.lower()) + r'\b', '', text)  # Remove word only if it's a whole word
#     text.strip()
# positive_score,negative_score = count_word_occurrences(text,word_dictionary)
# polarity_score = (positive_score - negative_score)/((positive_score + negative_score) + 0.000001)
# subjectivity_score = (positive_score + negative_score)/ ((len(text.split())) + 0.000001)
# complex_word_count = count_complex_words(text)
# print(complex_word_count)
# word_count = len(text.split())
# avg_sentence_length = round(average_sentence_length(para))
# complex_word_percent = (complex_word_count/word_count) * 100
# fog_index = 0.4 * (avg_sentence_length + complex_word_percent)
# syllable_per_word = round(count_average_syllables_in_word(text))
# personal_pronoun_count = count_personal_pronouns(para)
# word_length = round(calculate_average_word_length(text))
# avg_num_of_words_per_sentence = count_average_words_per_sentence(para)