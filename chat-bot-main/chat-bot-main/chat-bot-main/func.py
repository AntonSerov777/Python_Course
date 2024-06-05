import re

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords.words('russian')

from pymystem3 import Mystem
import torch
from catboost import CatBoostClassifier

from variables import TOKENIZER_BERT, MODEL_BERT


#Removing URL
def clean_url(review_text):
    return re.sub(r'http\S+', '', str(review_text))


#Removing all irrelevant characters (Numbers and Punctuation)
def clean_non_alphanumeric(review_text):
    return re.sub('[^а-яА-Я]', ' ', review_text)


#Convert all characters into lowercase
def clean_lowercase(review_text):
    return str(review_text).lower()


#Tokenization
def clean_tokenization(review_text):
    return word_tokenize(review_text)


#Removing Stopwords
def clean_stopwords(token):
    stop_words = set(stopwords.words('russian'))
    return [item for item in token if item not in stop_words]


def lemmatization(text):
    m = Mystem()
    lemmas = m.lemmatize(' '.join(text))
    return ' '.join(lemmas)


#Remove the words having length 
def clean_lenght(token):
    return [i for i in token.split() if len(i) > 2]


def convert_to_string(listReview):
    return ' '.join(listReview)


def preprocessing(text):
    clean_text = clean_url(text)         
    clean_text = clean_non_alphanumeric(clean_text)
    clean_text = clean_lowercase(clean_text)     
    clean_text = clean_tokenization(clean_text)
    clean_text = clean_stopwords(clean_text)
    clean_text = lemmatization(clean_text)
    clean_text = clean_lenght(clean_text)
    clean_text = convert_to_string(clean_text)
    return clean_text


def embed_bert_cls(text, MODEL_BERT, TOKENIZER_BERT):
    t = TOKENIZER_BERT(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = MODEL_BERT(**{k: v.to(MODEL_BERT.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    return embeddings[0].cpu().numpy()


def predict_intent(embed):
    model_cat_boost = CatBoostClassifier(iterations=150,
                                            depth=2,
                                            learning_rate=1,
                                            loss_function='MultiClass',
                                            verbose=True)
    model_cat_boost.load_model('model_cat_boost.pt')
    return model_cat_boost.predict(embed)[0]




