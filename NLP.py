###############################################################
# Title: Natural Language Processing
###############################################################

###############################################################
# Loading packages
###############################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
# import psycopg2 as ps
# import enchant


# Custom library
import Dictionary as cd
import textblob as TextBlob

################################################################
# Initialization
################################################################
# Constants


#################################################################
# Cleansing of Text Data
#################################################################


# Uniform Case sensitive
def caseChange(data, column):
    text_ind = data.columns.get_loc(column)
    text = data.iloc[:, text_ind]
    data['clean_text'] = text.apply(lambda x: ' '.join([word.lower() for word in x.split()]))
    return data


# Remove stopwords
def removeStopwords(data, column):
    # nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = stopwords.words("english")
    data[column] = data[column].fillna("")
    data[column] = data[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return data


# Spell Check
def spellCheck(data, column):
    import enchant
    data['clean_text'] = data['clean_text'].str.decode('utf-8')
    checker = enchant.Dict("en_US")
    data['clean_text'].map(lambda x: checker.suggest(data['clean_text']))
    return data


# Remove URLs
def removeURL(data, column):
    regURL = re.compile(r"http.?://[^\s]+[\s]?")
    data[column].replace(regURL, "", inplace=True)
    return data


# Remove user names
def removeUserName(data, column):
    regUser = re.compile(r"@[^\s]+[\s]?")
    data[column].replace(regUser, "", inplace=True)
    return data


# Remove numbers
def removeNumbers(data, column):
    regNum = re.compile(r"\s?[0-9]+\.?[0-9]*")
    data[column].replace(regNum, "", inplace=True)
    return data


# Remove special characters
def removeSpecialChar(data, column):
    for remove in map(lambda r: re.compile(re.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                           "@", "%", "^", "*", "(", ")", "{", "}",
                                                           "[", "]", "|", "/", "\\", ">", "<", "-",
                                                           "!", "?", ".", "'",
                                                           "--", "---", "#"]):
        data["clean_text"].replace(remove, "", inplace=True)
    return data


# Expand Contractions
def expandContraction(data, column):
    for i in range(0, data.shape[0]):
        data['clean_text'][i] = cd.expandContractions(data['clean_text'][i])
    return data


# Stemming
def stemming(data, column):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    data[column] = data[column].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
    return data


# Lemmazation
def lemmazation(data, column):
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    data[column] = data[column].apply(lambda x: ' '.join([lmtzr.lemmatize(word, 'v') for word in x.split()]))
    return data


# Parts of Speech Tagging (POS)
def pos(data, column):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    # print(data[column])
    data[column] = data[column].apply(lambda x: nltk.pos_tag(nltk.word_tokenize(x)))
    # print(data[column])
    return data


# Count of a word
def wordCount(data, column, word):
    data['count'] = np.zeros(data.shape[0], dtype=int)
    for i in range(0, data.shape[0]):
        data['count'][i] = sum(1 for match in re.finditer(word, data[column][i]))
    return data


# Remove word less than specific char
def removeWords(data, column, n):
    data[column] = data[column].apply(lambda x: ' '.join([w for w in x.split() if len(w) > n]))
    return data


# Remove specific Pattern of words                                          ## TO BE CHECKED
def removePattern(data, column, pattern):
    r = re.findall(pattern, data[column])
    for i in r:
        data[column] = re.sub(i, '', data[column])
    return data

# group words into list with specified length
def group(lst, n, chunk):
    if chunk == True:
        for i in range(0, len(lst), n):
            val = lst[i:i + n]
            if len(val) == n:
                if val[1] != 'O':  # Remove Out side of chunk at position 1
                    yield tuple(val)
    else:
        for i in range(0, len(lst), n):
            val = lst[i:i + n]
            if len(val) == n:
                yield tuple(val)


# Named-Entity Recognization
def namedEntityRecog(data, column):
    import nltk
    nltk.download('words')   # import words
    from nltk import word_tokenize, pos_tag, ne_chunk
    for i in range(0, data.shape[0]):
        data[column][i] = ne_chunk(pos_tag(word_tokenize(data[column][i])))  # chunk POS tags
        data[column][i] = nltk.chunk.tree2conllstr(data[column][i])          # Convert chunk tree to string
        data[column][i] = data[column][i].split()                            # Split string to words
        # Delete pos tags and rename
        del data[column][i][1::3]                                            # Delete pos tags as NN, NNP
        # data[column][i] = data[column][i]
        data[column][i] = list(group(data[column][i], 2, chunk=True))                    # Group them in list
    return data

# Extract specific tags                                                     ## TO BE CHECKED
def regExtract(data, column, regex):
    extracted = []
    # Loop over the words in the tweet
    for i in range(0, data.shape[0]):
        for j in data:
            word = re.findall(regex, j)
            extracted.append(word)
    return data


# Generate Word Cloud
def wordCloud(words, mask):
    # all_words = ' '.join([text for text in data['clean_text']])
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wordcloud = WordCloud(mask=mask, width=800, height=500, random_state=21, max_font_size=110).generate(words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    return plt


# Generate sentiment using textblob
def sentiment(data, column):
    from textblob import TextBlob
    data['polarity'] = None
    data['subjectivity'] = None
    data['sentiment'] = None

    for i in range(0, data.shape[0]):
        polarity = TextBlob(data[column][i]).sentiment.polarity
        data['polarity'][i] = polarity
        if polarity > 0:
            data['sentiment'][i] = 'positive'
        elif polarity < 0:
            data['sentiment'][i] = 'negative'
        else:
            data['sentiment'][i] = 'neutral'
        data['subjectivity'][i] = TextBlob(data[column][i]).sentiment.subjectivity
    return data

#################################################################
# Ends
#################################################################