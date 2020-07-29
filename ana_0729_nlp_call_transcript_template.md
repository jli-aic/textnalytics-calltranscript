This notebook listed a few reusable objects extracted from the full NLP call transcript use case
-----------

Make sure pip is up-to-date
-----------


```python
#pip install --upgrade pip
```

Prepare libraries
--


```python
pip install -U scikit-learn scipy matplotlib
```


```python
pip install textblob
```


```python
pip install wordcloud
```


```python
pip install gensim
```


```python
import pandas as pd 
import re
import string
import pickle
import sys

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

import collections
from collections import Counter

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from textblob import TextBlob

from gensim import matutils, models
import scipy.sparse
```

Import data and set the structure
---


```python
df = pd.read_csv('//file_direction/filename', sep='|')
df.shape
```


```python
df
```


```python
#Group the transcripts ("content") per Call ID ("call_id")
df=df.groupby(['call_id' ])['content'].apply(' '.join).reset_index() 
df.shape
```


```python
df
```

Cleaning 1 - RE for general grooming
---


```python
# Apply a first round of text cleaning techniques
#import re
#import string

def re_clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

r1 = lambda x: re_clean_text(x)
```


```python
#Calling re_clean_text function
data_clean = pd.DataFrame(df.content.apply(r1))
data_clean
```

#  Cleaning 2 - stemming
-----------------------------------
Steps to stem a document

1. take the doc as the input
2. read line by line
3. tokenize the line
4. stem the words
5. output the stemmed words

Use Cases:
sentimental analysis
document clustering
information retrieval




```python
#import nltk
#nltk.download('punkt')
```


```python
#A second round of cleanning technique - stemming
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.stem import PorterStemmer

st=PorterStemmer()

def stem_text(text):
    token_words=word_tokenize(text)
    stem_text=[]
    for word in token_words:
        stem_text.append(st.stem(word))
        stem_text.append(" ")
    return "".join(stem_text)

s1=lambda x: stem_text(x)
```


```python
# Let's take a look at the updated text (a longer run time, 30~ mins)
data_clean = pd.DataFrame(data_clean.content.apply(s1))
data_clean
```

# Cleaning 3 - removing Stop Word
_______________________________________________


```python
#import nltk
#from nltk.tokenize import word_tokenize 

#nltk.download('stopwords')
#from nltk.corpus import stopwords

# set of stop words
stop_words = set(stopwords.words('english')) 
```


```python
def rm_stopword(text):
    # tokens of words  
    token_words=word_tokenize(text)
    # define variable
    filtered_sentence = [] 
    
    for word in token_words:
        if word not in stop_words:
            filtered_sentence.append(word)
            filtered_sentence.append(" ")
    return "".join(filtered_sentence)

rsw=lambda x: rm_stopword(x)
```


```python
#Apply stop word removal
data_clean = pd.DataFrame(data_clean.content.apply(rsw))
data_clean 
```


```python
# To add the call id back to the dataframe
data_clean['call_id'] = df['call_id']
data_clean

```

Pickle the file to a local drive
-----------------------------


```python
#import pickle
```


```python
# Make a new directory (stage_Data/) to hold the intermediate pickle files
!mkdir stage_data
```


```python
# Pickle files for later use (to save it physically to local drive)
data_clean.to_pickle("stage_data/cust_data_clean.pkl")


#Read from pickle file to verify
#pd.read_pickle('stage_data/data_clean3.pkl')
```

Clearning 4 - finding Common Words
------------------


```python
#import collections
```

Define the functin of top Common Word Count per document (call) within the file
-------


```python
#import re
#from collections import Counter

def word_counts_line (text):
    counts = []
    #to convert the line of words into a list of words
    text = re.findall(r'\w+', text)
    
    #counts the number each time a word appears and get the top 30 common words
    counts = Counter(text).most_common(30) 
    return counts

cd = lambda x: word_counts_line(x)
 
```


```python
#Apply stop word removal
pd.DataFrame(data_clean.content.apply(cd))
```

Define the functin of top Common Word Count per file  
-------


```python
# Look at the 100 most common top words --> add them to the stop word list
#from collections import Counter

# To pull out 100 top words from each call
top100_word_cnt = Counter(" ".join(data_clean["content"]).split()).most_common(100)
top100_word_cnt
```


```python
add_stop_words = [word for word, count in top100_word_cnt if count > 500]
add_stop_words
```

Clearning 5 - Apply additional stop words
------------------


```python
len(stop_words)
```


```python
len(add_stop_words)
```


```python
#Apply additional stop word removal
stop_words = stop_words.union(add_stop_words)

data_clean = pd.DataFrame(data_clean.content.apply(rsw))
data_clean
```


```python
#adding the call id back to the datagrame
data_clean['call_id'] = df['call_id'] 
data_clean
```


```python
#Save the data to pickle file
data_clean.to_pickle("stage_data/data_clean.pkl")
```

Run Word Clouds
---------


```python
#To make some word clouds
#from wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)
```

Extract Noun function
-----------------------


```python
#import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
```


```python
# Let's create a function to pull out nouns from a string of text
#from nltk import word_tokenize, pos_tag

def nouns(text):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)
```

Prepare stop words
---------


```python
# iterate through the csv file 
from wordcloud import WordCloud, STOPWORDS 

comment_words = '' 
stopwords = set(STOPWORDS) 

stopwords = stopwords.union(add_stop_words)
stopwords = stopwords.union(stop_words)
```


```python
for val in data_clean.content: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value      
    comment_words += " ".join(val.split() )+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
```


```python
#Apply Noun extraction function to failed si
text_nouns = pd.DataFrame(data_clean.content.apply(nouns))
text_nouns
```


```python
#Adding call id back to the dataframe 
text_nouns['call_id'] = data_clean['call_id']
text_nouns
```


```python
text_nouns.to_pickle("stage_data/text_nouns.pkl")
```

Sentiment Analysis
---


```python
#import sys
#from textblob import TextBlob
```


```python
# Create lambda functions to find the polarity and subjectivity per call
p = lambda x: TextBlob(x).sentiment.polarity
s= lambda x: TextBlob(x).sentiment.subjectivity

data_clean['polarity'] = data_clean['content'].apply(p)
data_clean['subjectivity'] = data_clean['content'].apply(s)
data_clean
```


```python
data_clean.describe()
```


```python
data_clean.polarity[data_clean.polarity< 0].sum()
```

Topic Modeling - Attempt #1 (All Text)
-----------------------------------------


```python
df=pd.read_pickle("stage_data/data_clean.pkl")
```


```python
#import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(df.content)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df )
```


```python
#pip install gensim
```


```python
# Import the necessary modules for LDA with gensim
#from gensim import matutils, models
#import scipy.sparse
```


```python
# One of the required inputs is a term-document matrix
tdm = df.transpose()
tdm.head()
```


```python
#Transforming the term-document matrix into corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
```


```python
#Further format prep for Gensim
vec
id2word = dict((v, k) for k, v in vec.vocabulary_.items())
```


```python
#To define two other parameters - first try 2 topics and 10 passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)
lda.print_topics()
```


```python
#Then try 30 topics and 10 passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=30, passes=10)
lda.print_topics()
```


```python
#Then try 100 topics and 10 passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=100, passes=10)
lda.print_topics()
```

Topic Modeling - Attempt #2 (Nouns Only)
------------------------------------------


```python
df=pd.read_pickle("stage_data/text_nouns.pkl")
df
```


```python
#import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(df.content)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df )
```


```python
tdm = df.transpose()
tdm.head()
```


```python
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
```


```python
vec
id2word = dict((v, k) for k, v in vec.vocabulary_.items())
```


```python
#Try 30 topics and 100 passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=30, passes=100)
lda.print_topics()
```

Topic Modeling - potentially attemps may be including adjectives also, topic modeling per each lauguage, etc.
--------------------------------------------------

Clean up the memory (misc.)
------


```python
df.info(memory_usage='deep')
```


```python
import gc
```


```python
%whos DataFrame
```


```python
del [[df]]
gc.collect()
df=pd.DataFrame()
 
```


```python

```
