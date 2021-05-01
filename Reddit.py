import praw
import pandas as pd
import string
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import numpy  as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas. plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow_text 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


########## Scraping  data cleaning
reddit = praw.Reddit(client_id='Wiv9jPFUOo3klw', \
                     client_secret='_sbHKx0nqWdYt_4ZIY75Yb8SPmZ-Xg', \
                     user_agent='Seif_Mejri', \
                     username='MrSifon98 ', \
                     password='01041998')
      
subreddit = reddit.subreddit('Nootropics')
top_subreddit = subreddit.top(limit=100000)
for submission in subreddit.top(limit=10000000):
    print(submission.title, submission.id)
submission = reddit.submission(url="https://www.reddit.com/r/politics/comments/jptq5n/megathread_joe_biden_projected_to_defeat/")
submission = reddit.submission(url="https://www.reddit.com/r/news/comments/jptqj9/joe_biden_elected_president_of_the_united_states/")

R_comments = pd.read_excel(r'C:\Users\Administrator\Desktop\data2.xlsx')
a=[]
a= R_comments['comments']
a=list(a)
submission.comments.replace_more(limit=1300000)
for comment in submission.comments.list():
    a.append(comment.body)



########## Data cleaning & Tokenization
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
     # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return text

b=[]
for comment in a :
    b.append(clean_text(comment))
        
Rreddit_comments=pd.DataFrame({'comments':b})
Rreddit_comments.to_excel(r'C:\Users\Administrator\Desktop\data2.xlsx')



##########Sentiment Analysis(polarity & compound value)

sia = SIA()
results = []
for comment in b:
    pol_score = sia.polarity_scores(comment)
    pol_score['headline'] = comment
    results.append(pol_score)

sentiments_dataset = pd.DataFrame(results)



sentiments_dataset["review_type"] = sentiments_dataset["compound"].apply(
  lambda x :"bad" if x <0 else ("good" if x>0 else "neutral") 
  )

good_reviews = sentiments_dataset[sentiments_dataset.review_type == "good"]
bad_reviews = sentiments_dataset[sentiments_dataset.review_type == "bad"]
neutral_reviews = sentiments_dataset[sentiments_dataset.review_type == "neutral"]

good_df = good_reviews.sample(n=len(neutral_reviews), random_state=RANDOM_SEED)
bad_df = bad_reviews.sample(n=len(neutral_reviews), random_state=RANDOM_SEED)
neutral_df=neutral_reviews

review_df = good_df.append(bad_df).append(neutral_df).reset_index(drop=True)
print(review_df.shape)

review_df = review_df[['headline','review_type']]
review_df.columns = ['review', 'review_type']

############
######### Exploring the review
good_reviews_text = " ".join(good_reviews.headline.to_numpy().tolist())
bad_reviews_text = " ".join(bad_reviews.headline.to_numpy().tolist())
good_reviews_cloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(good_reviews_text)
bad_reviews_cloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(bad_reviews_text)

def show_word_cloud(cloud, title):
  plt.figure(figsize=(20,10))
  plt.imshow(cloud, interpolation='bilinear')
  plt.title(title)
  plt.axis("off")
  plt.show()
  plt.savefig(title+'png', dpi=300)


show_word_cloud(good_reviews_cloud, "Good reviews common words")
show_word_cloud(bad_reviews_cloud, "Bad reviews common words")




##########Vecorizing Text
type_one_hot = OneHotEncoder(sparse=False).fit_transform(
  review_df.review_type.to_numpy().reshape(-1, 1))


train_reviews, test_reviews, y_train, y_test =\
  train_test_split(
    review_df.review,
    type_one_hot,
    test_size=.1,
    random_state=RANDOM_SEED
  )


X_train = []
for r in tqdm(train_reviews):
  emb = use(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_train.append(review_emb)
X_train = np.array(X_train)
X_test = []
for r in tqdm(test_reviews):
  emb = use(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_test.append(review_emb)
X_test = np.array(X_test)
print(X_train.shape, y_train.shape)


##################### Keras Model
model = keras.Sequential()
model.add(keras.layers.Dense(units=256,input_shape=(X_train.shape[1], ), activation='relu' ))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128,activation='relu'))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(3, activation='softmax'))

#### Compiling and fitting
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])

history = model.fit(X_train, y_train,epochs=10,batch_size=16,validation_split=0.1,verbose=1,
                    shuffle=True)
 


#### plotting for optimization
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Cross-entropy loss")
plt.figure(figsize=(40,20))
plt.legend()
plt.savefig('5.png', dpi=300)

plt.style.use('dark_background')
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.ylim(ymin=0, ymax=1)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig('6.png', dpi=300)



#### evaluating the model
model.evaluate(X_test, y_test)





##### Some Manual tests and prediction
print(test_reviews.iloc[0])
print("Bad" if y_test[0][0] == 1 else "Good")

y_pred = model.predict(X_test[:1])
print(y_pred)
"Bad" if np.argmax(y_pred) == 0 else "neutral"

print(test_reviews.iloc[297])
print("Bad" if y_test[0][0] == 1 else "Good")

print(test_reviews.iloc[1])
print("Bad" if y_test[1][0] == 1 else "Good")

y_pred = model.predict(X_test[1:2])
print(y_pred)
"Bad" if np.argmax(y_pred) == 0 else "Good"



