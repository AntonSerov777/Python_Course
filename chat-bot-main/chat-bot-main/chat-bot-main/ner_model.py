import pandas as pd
import numpy as np
from tqdm import trange
import ast
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from seqeval.metrics import f1_score, accuracy_score

from variables import FILE_NER, MAX_LEN, BS, EPOCHS_NER, MAX_GRAD_NORM, TOKENIZER, EPOCHS



class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                           s["label"].values.tolist())]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        

def read_data(path):
    data = pd.read_csv(path, sep='\t')
    data_new = pd.DataFrame(columns = ['sentence', 'word', 'label'])
    sentence_list, word_list, label_list = [], [], []
    for i, row in data.iterrows():
      for j in range(len(ast.literal_eval(row.label))):
        sentence_list.append(i+1)
        word_list.append(ast.literal_eval(data.label[i])[j]['text'])
        label_list.append(ast.literal_eval(data.label[i])[j]['labels'][0])
    data_new['sentence'] = sentence_list
    data_new['word'] = word_list
    data_new['label'] = label_list
    return data_new


def tokenize_and_preserve_labels(sentence, text_labels, TOKENIZER):
    tokenized_sentence = []
    labels = []
    print('Предложение:',  sentence)
    for word, label in zip(sentence, text_labels):
        tokenized_word = TOKENIZER.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    return tokenized_sentence, labels


    
def split_to_train_test(input_ids, tags, attention_masks):
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=0, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=0, test_size=0.1)
    
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)
    
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BS)
    
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BS)
        
    return train_dataloader, valid_dataloader
    

def bert_fine_tuning(model, train_dataloader, valid_dataloader, EPOCHS_NER, MAX_GRAD_NORM, tag_values):
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )
    
    total_steps = len(train_dataloader) * EPOCHS_NER

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # ========================================
    #          FINE TUNING NER MODEL
    # ========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_values, validation_loss_values = [], []
    validation_accuracy_values = []
    validation_f1_values = []
    
    for _ in trange(EPOCHS_NER, desc="Epoch"):
        model.train()
        total_loss = 0
    
        # Training loop
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            b_input_ids = b_input_ids.to(torch.long)
            b_input_mask = b_input_mask.to(torch.long)
            b_labels = b_labels.to(torch.long)
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
    
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
    
        loss_values.append(avg_train_loss)
        
        # Validation    
        model.eval()
        eval_loss = 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(torch.long)
            b_input_mask = b_input_mask.to(torch.long)
            b_labels = b_labels.to(torch.long)

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)
    
        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if tag_values[l_i] != "PAD"]
        y_true, y_pred = [], []
        y_true.append(valid_tags)   
        y_pred.append(pred_tags)                           
        print("Validation Accuracy: {}".format(accuracy_score(y_true, y_pred)))
        validation_accuracy_values.append(accuracy_score(y_true, y_pred))
        print("Validation F1-Score: {}".format(f1_score(y_true, y_pred)))
        validation_f1_values.append(f1_score(y_true, y_pred))
        print()
        
    return model 
    
    
def get_model(data_new):
    
    getter = SentenceGetter(data_new)

    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    labels = [[s[1] for s in sentence] for sentence in getter.sentences]
    tag_values = list(set(data_new["label"].values))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs, TOKENIZER)
            for sent, labs in zip(sentences, labels)
        ]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    
    tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs, TOKENIZER) \
                                  for sent, labs in zip(sentences, labels)]
    
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    input_ids = pad_sequences([TOKENIZER.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", value=0.0,
                                  truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                             maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                             dtype="long", truncating="post")
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
    
    train_dataloader, valid_dataloader = split_to_train_test(input_ids, tags, attention_masks)
    model = BertForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )
    model = bert_fine_tuning(model, train_dataloader, valid_dataloader, EPOCHS, MAX_GRAD_NORM, tag_values)
    
    return model, tag_values


def make_prediction(test_sentence, TOKENIZER, model, tag_values):
    tokenized_sentence = TOKENIZER.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)  
    tokens = TOKENIZER.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    result_dict = {}
    for label1 in tag_values:
        result_dict[label1] = []
    for i, label2 in enumerate(new_labels):
        result_dict[label2].append(new_tokens[i])

    return result_dict


if __name__ == '__main__':
    data_new = read_data(FILE_NER)
    model, tag_values = get_model(data_new)
    torch.save(model, "ner_model.pt")
    with open('tag_values.pkl', 'wb') as f:
        pickle.dump(tag_values, f)


