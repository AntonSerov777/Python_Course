import numpy as np  
import pandas as pd
import torch
from func import preprocessing, embed_bert_cls, predict_intent
from scipy.spatial import distance
from ner_model import make_prediction
from fit_embed_and_model import transform_embed, Net
from variables import THRESH, ANSWER, INTENT, NEUTRAL_RESPONSE, MODEL_NER_PATH, \
    TAG_VALUES, Y_DICT, TOKENIZER, TOKENIZER_BERT, MODEL_BERT


def get_prepared_answer(clean_text_new, data):
    if len(data[data.clean_text == clean_text_new]) > 0:
        return data[data.clean_text == clean_text_new].answer.iat[0]
    else:
        return 0


with open('pattern_data.pkl', 'rb') as f:
    pattern_data = pd.read_pickle(f)
with open('full_data.pkl', 'rb') as f:
    data = pd.read_pickle(f)
    
print('Введите ваш вопрос к боту:')
while ANSWER != NEUTRAL_RESPONSE and INTENT != 2:
    text = input()
    clean_text_new = preprocessing(text)
    #смотреть, нет ли именно такого вопроса в data
    answer = get_prepared_answer(clean_text_new, pattern_data)
    if answer != 0:
        print(answer)
        continue
    
    text_embed = embed_bert_cls(clean_text_new, MODEL_BERT, TOKENIZER_BERT) #bert embed
    siam_embed = transform_embed(text_embed) #siam embed    
    intent = predict_intent(siam_embed)
    print(intent)
    intent = list(Y_DICT.keys())[list(Y_DICT.values()).index(intent)]
    print(intent)
    
    with open('embed_X.npy', 'rb') as f:
        embed_X = np.load(f) #get embedings
        
    dist_list = []
    for idx, _ in data[data.category == intent].iterrows():        
        dst = distance.euclidean(embed_X[idx, :], siam_embed)
        dist_list.append(dst)
    result_dict = None
    
    # если вопрос не похож ни на один из обучающей выборки
    if min(dist_list) > THRESH:
        model_ner = torch.load(MODEL_NER_PATH)
        result_dict_ner = make_prediction(text, TOKENIZER, model_ner, TAG_VALUES)
        # дальше перевод на консультанта
        print('Запрос не совсем понятен.', NEUTRAL_RESPONSE)
        print('*Информация, которая поступает консультанту:')
        print(result_dict_ner)
    else:
        intent_df = data[data.category == intent].reset_index()
        answer = intent_df[intent_df.index == np.argmin(dist_list)].answer.values[0]
        print(answer)

