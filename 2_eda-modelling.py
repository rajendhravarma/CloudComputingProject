#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

pd.set_option('max_colwidth', 500)
sns.set_style("whitegrid")


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from nltk.tokenize import ToktokTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

# To save the trained model on local storage
import pickle
from joblib import dump, load


# In[ ]:


def LemmatizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def RemoveStopWords(text):
    stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]
    words = text.split(' ')
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))


# # Read Dataset

# In[ ]:


questions = pd.read_csv("/home/aditya_bagad2/cleaned_questions.csv")
tags = pd.read_csv("/home/aditya_bagad2/tags.csv")


# In[ ]:


questions.head()


# In[ ]:


tags.head()


# # EDA

# ## Most Used Tag

# In[ ]:


top10_tags = pd.DataFrame(tags.value_counts(['tag'], sort=True)[:10]).reset_index().rename(columns={0: 'tag_count'})

plt.subplots(figsize=(10, 5))
sns.barplot(data=top10_tags, x="tag", y="tag_count", color="lightblue")
plt.title("Top 10 Most Used Tags")

plt.show()


# ## Top 10 Most Upvoted Questions 

# In[ ]:


top10_upvoted = questions.sort_values(by='score', ascending=False)[:10][['title', 'score']]

top10_upvoted


# ## Number of Questions Posted Over Years

# In[ ]:


questions['creationdate'] = pd.to_datetime(questions['creationdate'])

questions['year'] = questions['creationdate'].map(lambda x: x.year)

que_posted = pd.DataFrame(questions.value_counts(['year'], ascending=True)).reset_index().rename(columns={0: 'questions count'})


# In[ ]:


plt.subplots(figsize=(10, 5))
sns.lineplot(data=que_posted, x="year", y="questions count")
plt.title("Number of Questions Posted Over Years")

plt.show()


# # Joining the Data

# In[ ]:


questions.drop(columns=['creationdate', 'score'], inplace=True)

questions.dropna(subset=['title'], inplace=True)

# tags.dropna(subset=['tag'], inplace=True)


# In[ ]:


# grouped_tags = pd.DataFrame(tags.groupby("id")['tag'].apply(lambda tags: ' '.join(tags))).reset_index()
# grouped_tags


# In[ ]:


questions['id'] = questions['id'].astype(int)
tags['id'] = tags['id'].astype(int)
# grouped_tags['id'] = grouped_tags['id'].astype(int)


# In[ ]:


# Remove StopWords
questions['title'] = questions['title'].apply(lambda x: RemoveStopWords(x))


# In[ ]:


# final_dataset = pd.merge(questions, grouped_tags, on=['id'], how='left')
final_dataset = pd.merge(questions, tags, on=['id'], how='left')


# In[ ]:


final_dataset.shape


# In[ ]:


final_dataset.head()


# In[ ]:


# # Save Cleaned Dataset
final_dataset.to_csv("/home/aditya_bagad2/cleaned_dataset_v2.csv", index=False, header=True)


# In[ ]:


final_dataset2 = pd.merge(questions, tags, on=['id'], how='left')


# In[ ]:


unique_tags = list(set(final_dataset.tag.dropna().to_list()))


# In[ ]:


with open("/home/aditya_bagad2/unique_tags", "wb") as fp:
    pickle.dump(unique_tags, fp)

with open("/home/aditya_bagad2/unique_tags", "rb") as fp:
    b = pickle.load(fp)


# In[ ]:


len(b)


# In[ ]:


unique_tags_dict = {i:b[i] for i in range(len(b))}


# In[ ]:


unique_tags_dict


# In[ ]:


with open("/home/aditya_bagad2/unique_tags_dict", "wb") as fp:
    pickle.dump(unique_tags_dict, fp)

with open("/home/aditya_bagad2/unique_tags_dict", "rb") as fp:
    c = pickle.load(fp)


# # Modelling

# In[ ]:


final_dataset = pd.read_csv("/home/aditya_bagad2/cleaned_dataset_v2.csv")


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset.dropna(subset=['tag'], inplace=True)
final_dataset.dropna(subset=['title'], inplace=True)


# In[ ]:


inv_map = {v: k for k, v in c.items()}

final_dataset['tag_id'] = final_dataset['tag'].map(lambda x: inv_map[x])

final_dataset['tag_id'] = final_dataset['tag_id'].astype(int)


# In[ ]:


# X1 = final_dataset['title']
X1 = final_dataset['title'][:50000]


# In[ ]:


vectorizer_X1 = TfidfVectorizer(analyzer = 'word')

X1_tfidf = vectorizer_X1.fit_transform(X1)


# In[ ]:


dump(vectorizer_X1, '/home/aditya_bagad2/Vocab_v1.joblib')


# In[ ]:


y = final_dataset['tag_id'][:50000].astype(int)


# In[ ]:


# multilabel_binarizer = MultiLabelBinarizer()

# y_bin = multilabel_binarizer.fit_transform(y)


# In[ ]:


X1_tfidf.shape, y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X1_tfidf, y, test_size = 0.2, random_state = 0)


# ## Logistic Regression

# In[ ]:


classifier = LogisticRegression()


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)

y_pred


# In[ ]:


# Save model for Logistic Regression
dump(clf, '/home/aditya_bagad2/LogisticRegression_Modelv2.joblib')


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


# joblib.dump(classifier_model, 'LogisticRegression_Modelv2.joblib')

