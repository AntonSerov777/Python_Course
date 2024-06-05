import pandas as pd
import numpy as np  
from catboost import CatBoostClassifier
import json 
from func import preprocessing, embed_bert_cls
from variables import Y_DICT, TOKENIZER_BERT, MODEL_BERT, DATA_PATH


#=================FUNCTIONS FOR FIT CAT BOOST MODEL====================#

def load_prepared_json(data_path):
    f = open(data_path, encoding="utf8")
    data = json.load(f)
    df_l = []
    for tag in data['ourIntents']:
      for pattern in tag['patterns']:
        df_l.append([tag['tag'], 
                     pattern['pattern'][0], 
                     pattern['response'][0]])
        
    data = pd.DataFrame(np.array(df_l), 
                        columns = ['category', 'question', 'answer'])    
    return data
    

def get_data(data_path, model_bert, y_dict = Y_DICT):   
    data = load_prepared_json(data_path) #сохраняем данные с приветствиями и прощаниями
    data['clean_text'] = data['question'].apply(lambda x: preprocessing(x))
    pattern_data = data.copy()
    data = data[(data.category != 'приветствие') & \
                  (data.category != 'прощание') & \
                  (data.category != 'имя')].reset_index().drop('index', axis=1)
    
    data['embed'] = data.clean_text.apply(lambda x: embed_bert_cls(x, model_bert, TOKENIZER_BERT))
    
    texts_list = []
    for i in range(data['clean_text'].shape[0]):
      texts_list.append(embed_bert_cls(data['clean_text'][i], 
                        model_bert, 
                        TOKENIZER_BERT))
      
    X = np.array(texts_list)
    data['category_num'] = data['category'].apply(lambda x: y_dict[x])
    y = np.array(data['category_num'])
    
    data.to_pickle("full_data.pkl")
    #отдельно сохраняем файл в котором есть приветствие и прощание
    pattern_data.to_pickle("pattern_data.pkl") 
    return data, X, y


def fit_catboost_model(X, y):
    model = CatBoostClassifier(iterations=300,
                               depth=2,
                               learning_rate=1,
                               loss_function='MultiClass',
                               verbose=True)
    model.fit(X, y)
    model.save_model('model_cat_boost.pt',
                               format="cbm",
                               export_parameters=None,
                               pool=None)
    return model


if __name__ == '__main__':
    data, X, y = get_data(DATA_PATH, MODEL_BERT)
    embed_X = fit_tiplet_loss_embed(X, y)
    with open('embed_X.npy', 'wb') as f:
        np.save(f, embed_X) #save embedings
    # with open('embed_X.npy', 'rb') as f:
    #     embed_X = np.load(f) #get embedings
    fit_catboost_model(embed_X, y)
