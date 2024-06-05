import pickle
from transformers import BertTokenizer, AutoTokenizer, AutoModel


THRESH = 0.4
ANSWER = 'Привет!'
INTENT = 1
NEUTRAL_RESPONSE = 'Перенаправляю вас к консультанту'
MODEL_NER_PATH = "ner_model.pt"

with open('tag_values.pkl', 'rb') as f:
    TAG_VALUES = pickle.load(f)
    
Y_DICT = {'личные_документы': 0,
          'финансы, налоги, штрафы': 1,
          'семья': 2,
          'работа': 3, 
          'транспорт': 4}

FILE_NER = 'project-4-at-2023-03-28-12-16-87b28895.csv' 
DATA_PATH = 'data.json'
LR = 0.01
EPOCHS = 1500
MAX_LEN = 75
BS = 32
EPOCHS_NER = 50
MAX_GRAD_NORM = 1.0

TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                          do_lower_case=False)

TOKENIZER_BERT = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
MODEL_BERT = AutoModel.from_pretrained("cointegrated/rubert-tiny")
