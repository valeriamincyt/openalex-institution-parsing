#003_model_testing
# %%
import os
import re
import json
import boto3
import pickle
from unidecode import unidecode
from collections import Counter
from langdetect import detect
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
from datetime import datetime
import time

# %%
# Define the path
base_save_path = "./"
rutaDatos = "../Datos/"
curr_model_artifacts_location = f"{rutaDatos}institution_tagger_v2_artifacts/"
institution_gold_datasets = f"{rutaDatos}institution_gold_datasets_v2/"

# Load the needed files
with open(os.path.join(curr_model_artifacts_location, "departments_list.pkl"), "rb") as f:
    departments_list = pickle.load(f)

print("Loaded list of departments")

with open(os.path.join(curr_model_artifacts_location, "full_affiliation_dict.pkl"), "rb") as f:
    full_affiliation_dict = pickle.load(f)

print("Loaded affiliation dictionary")

with open(os.path.join(curr_model_artifacts_location, "multi_inst_names_ids.pkl"), "rb") as f:
    multi_inst_names_ids = pickle.load(f)
    
print("Loaded list of institutions that have common name with other institutions.")

with open(os.path.join(curr_model_artifacts_location, "countries_list_flat.pkl"), "rb") as f:
    countries_list_flat = pickle.load(f)

print("Loaded flat list of countries")

with open(os.path.join(curr_model_artifacts_location, "countries.json"), "r") as f:
    countries_dict = json.load(f)

print("Loaded countries dictionary")

with open(os.path.join(curr_model_artifacts_location, "city_country_list.pkl"), "rb") as f:
    city_country_list = pickle.load(f)

print("Loaded strings of city/country combinations")

with open(os.path.join(base_save_path, "affiliation_vocab.pkl"), "rb") as f:
    affiliation_vocab = pickle.load(f)

print('affiliation_vocab: ------------------------------')
print(len(affiliation_vocab.items()))
print(affiliation_vocab.items()[0:5])

inverse_affiliation_vocab = {i:j for j,i in affiliation_vocab.items()}

print("Loaded affiliation vocab")

# Load the tokenizers
#language_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", return_tensors='tf')
#language_tokenizer = DistilBertTokenizer.from_pretrained(f"{base_save_path}distilbert-local", return_tensors='tf')
language_tokenizer = DistilBertTokenizer.from_pretrained(f"{curr_model_artifacts_location}language_model/", return_tensors='tf')

data_collator = DataCollatorWithPadding(tokenizer=language_tokenizer, 
                                        return_tensors='tf')

#f"{rutaDatos}
basic_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(curr_model_artifacts_location, "basic_model_tokenizer"))

# Load the models
#language_model = TFAutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "language_model"))

language_model = TFAutoModelForSequenceClassification.from_pretrained(os.path.join(curr_model_artifacts_location, "language_model"))

language_model.trainable = False

basic_model = tf.keras.models.load_model(os.path.join(curr_model_artifacts_location, "basic_model"), compile=False)
basic_model.trainable = False

def string_match_clean(text):
    #replace "&" with "and"
    if "r&d" not in text.lower():
        text = text.replace(" & ", " and ")
        
    # take country out
    if text.strip().endswith(")"):
        for country in countries_list_flat:
            if text.strip().endswith(f"({country})"):
                text = text.replace(f"({country})", "")
        
    # use unidecode
    text = unidecode(text.strip())
    
    # replacing common abbreviations
    text = text.replace("Univ.", "University")
    text = text.replace("Lab.", "Laboratory")
    text = text.replace("U.S. Army", "United States Army")
    text = text.replace("U.S. Navy", "United States Navy")
    text = text.replace("U.S. Air Force", "United States Air Force")
    
    # take out spaces, commas, dashes, periods, etcs
    text = re.sub("[^0-9a-zA-Z]", "", text)
    
    return text

def get_country_in_string(text):
    """
    Looks for countries in the affiliation string to be used in filtering later on.
    """
    countries_in_string = []
    _ = [countries_in_string.append(x) for x,y in countries_dict.items() if 
         np.max([1 if re.search(fr"\b{i}\b", text) else 0 for i in y]) > 0]
    _ = [countries_in_string.append(x) for x,y in countries_dict.items() if 
         np.max([1 if re.search(fr"\b{i}\b", text.replace(".","")) else 0 for i in y]) > 0]
    return list(set(countries_in_string))

def max_len_and_pad(tok_sent):
    """
    Processes the basic model data to the correct input length.
    """
    max_len = 128
    tok_sent = tok_sent[:max_len]
    tok_sent = tok_sent + [0]*(max_len - len(tok_sent))
    return tok_sent


def get_language(orig_aff_string):
    """
    Guesses the language of the affiliation string to be used for filtering later.
    """
    try:
        string_lang = detect(orig_aff_string)
    except:
        string_lang = 'en'
        
    return string_lang

def get_initial_pred(orig_aff_string, string_lang, countries_in_string, comma_split_len):
    """
    Initial hard-coded filtering of the affiliation text to ensure that meaningless strings
    and strings in other languages are not given an institution.
    """
    if string_lang in ['fa','ko','zh-cn','zh-tw','ja','uk','ru','vi','ar']:
        init_pred = None
    elif len(string_match_clean(str(orig_aff_string))) <=2:
        init_pred = None
    elif ((orig_aff_string.startswith("Dep") | 
           orig_aff_string.startswith("School") | 
           orig_aff_string.startswith("Ministry")) & 
          (comma_split_len < 2) & 
          (not countries_in_string)):
        init_pred = None
    elif orig_aff_string in departments_list:
        init_pred = None
    elif string_match_clean(str(orig_aff_string).strip()) in city_country_list:
        init_pred = None
    elif re.search(r"\b(LIANG|YANG|LIU|XIE|JIA|ZHANG)\b", 
                   orig_aff_string):
        for inst_name in ["Hospital","University","School","Academy","Institute",
                          "Ministry","Laboratory","College"]:
            if inst_name in str(orig_aff_string):
                init_pred = 0
                break
            else:
                init_pred = None
                
    elif re.search(r"\b(et al)\b", orig_aff_string):
        if str(orig_aff_string).strip().endswith('et al'):
            init_pred = None
        else:
            init_pred = 0
    else:
        init_pred = 0
    return init_pred

def get_language_model_prediction(decoded_text, all_countries):
    """
    Preprocesses the decoded text and gets the output labels and scores for the language model.
    """
    lang_tok_data = language_tokenizer(decoded_text, truncation=True, padding=True, max_length=512)
    
    data = data_collator(lang_tok_data)
    all_scores, all_labels = tf.math.top_k(tf.nn.softmax(
            language_model.predict([data['input_ids'], 
                                    data['attention_mask']]).logits).numpy(), 20)
    
    all_scores = all_scores.numpy().tolist()
    all_labels = all_labels.numpy().tolist()
    print('all_labels: -------------------------------------')
    print(len(all_labels))
    print(all_labels[-10:])
    
    final_preds_scores = []
    for scores, labels, countries in zip(all_scores, all_labels, all_countries):
        final_pred, final_score, mapping = get_final_basic_or_language_model_pred(scores, labels, countries,
                                                                         affiliation_vocab, 
                                                                         inverse_affiliation_vocab)
        final_preds_scores.append([final_pred, final_score, mapping])
    
    return final_preds_scores

def get_basic_model_prediction(decoded_text, all_countries):
    """
    Preprocesses the decoded text and gets the output labels and scores for the basic model.
    """
    basic_tok_data = basic_tokenizer(decoded_text)['input_ids']
    basic_tok_data = [max_len_and_pad(x) for x in basic_tok_data]
    basic_tok_tensor = tf.convert_to_tensor(basic_tok_data, dtype=tf.int64)
    all_scores, all_labels = tf.math.top_k(basic_model.predict(basic_tok_data), 20)
    
    all_scores = all_scores.numpy().tolist()
    all_labels = all_labels.numpy().tolist()
    
    final_preds_scores = []
    for scores, labels, countries in zip(all_scores, all_labels, all_countries):
        final_pred, final_score, mapping = get_final_basic_or_language_model_pred(scores, labels, countries,
                                                                         affiliation_vocab, 
                                                                         inverse_affiliation_vocab)
        final_preds_scores.append([final_pred, final_score, mapping])
    
    return final_preds_scores


def get_final_basic_or_language_model_pred(scores, labels, countries, vocab, inv_vocab):
    """
    Takes the scores and labels from either model and performs a quick country matching
    to see if the country found in the string can be matched to the country of the
    predicted institution.
    """
    mapped_labels = [inv_vocab[i] for i,j in zip(labels,scores) if i!=vocab[-1]]
    mapped_labels = [inv_vocab[i] for i,j in zip(labels,scores)]
    scores = [j for i,j in zip(labels,scores) if i!=vocab[-1]]
    scores = [j for i,j in zip(labels,scores)]
    final_pred = mapped_labels[0]
    final_score = scores[0]
    if not full_affiliation_dict[mapped_labels[0]]['country']:
        pass
    else:
        if not countries:
            pass
        else:
            for pred,score in zip(mapped_labels, scores):
                if not full_affiliation_dict[pred]['country']:
                    # trying pass instead of break to give time to find the correct country
                    pass
                elif full_affiliation_dict[pred]['country'] in countries:
                    final_pred = pred
                    final_score = score
                    break
                else:
                    pass
    return final_pred, final_score, mapped_labels
    
def get_similar_preds_to_remove(decoded_string, curr_preds):
    """
    Looks for organizations with similar/matching names and only predicts for one of those organizations.
    """
    preds_to_remove = []
    pred_display_names = [full_affiliation_dict[i]['display_name'] for i in curr_preds]
    counts_of_preds = Counter(pred_display_names)
    
    preds_array = np.array(curr_preds)
    preds_names_array = np.array(pred_display_names)
    
    for pred_name in counts_of_preds.items():
        temp_preds_to_remove = []
        to_use = []
        if pred_name[1] > 1:
            list_to_check = preds_array[preds_names_array == pred_name[0]].tolist()
            for pred in list_to_check:
                if string_match_clean(full_affiliation_dict[pred]['city']) in decoded_string:
                    to_use.append(pred)
                else:
                    temp_preds_to_remove.append(pred)
            if not to_use:
                to_use = temp_preds_to_remove[0]
                preds_to_remove += temp_preds_to_remove[1:]
            else:
                preds_to_remove += temp_preds_to_remove
        else:
            pass
    
    return preds_to_remove  


def check_for_city_and_country_in_string(raw_sentence, countries, aff_dict_entry):
    """
    Checks for city and country and string for a common name institution.
    """
    if (aff_dict_entry['country'] in countries) & (aff_dict_entry['city'] in raw_sentence):
        return True
    else:
        return False


def get_final_prediction(basic_pred_score, lang_pred_score, countries, raw_sentence, lang_thresh, basic_thresh):
    """
    Performs the model comparison and filtering to get the final prediction.
    """
    
    # Getting the individual preds and scores for both models
    pred_lang, score_lang, mapped_lang = lang_pred_score
    pred_basic, score_basic, mapped_basic = basic_pred_score
    
    print('Individual preds and scores for both models: -----------------------------')
    print(f"lang: {pred_lang} - {score_lang}")
    print(f"basic: {pred_basic} - {score_basic}")
    
    # Logic for combining the two models
    
    final_preds = []
    final_scores = []
    final_cats = []
    check_pred = []
    if pred_lang == pred_basic:
        final_preds.append(pred_lang)
        final_scores.append(score_lang)
        final_cats.append('model_match')
        check_pred.append(pred_lang)
    elif score_basic > basic_thresh:
        final_preds.append(pred_basic)
        final_scores.append(score_basic)
        final_cats.append('basic_thresh')
        check_pred.append(pred_basic)
    elif score_lang > lang_thresh:
        final_preds.append(pred_lang)
        final_scores.append(score_lang)
        final_cats.append('lang_thresh')
        check_pred.append(pred_lang)
    elif (score_basic > 0.01) & ('China' in countries) & ('Natural Resource' in raw_sentence):
        final_preds.append(pred_basic)
        final_scores.append(score_basic)
        final_cats.append('basic_thresh_second')
        check_pred.append(pred_basic)
    else:
        final_preds.append(-1)
        final_scores.append(0.0)
        final_cats.append('nothing')
        
    print(final_preds)
    all_mapped = list(set(mapped_lang + mapped_basic))
    
    decoded_affiliation_string = string_match_clean(raw_sentence)
    all_mapped_strings = [full_affiliation_dict[i]['final_names'] for i in all_mapped]
          
    
    matched_preds = []
    matched_strings = []
    print('raw_sentence, decoded_affiliation_string, countries: -----------------------------')
    print(f"RAW: {raw_sentence}")
    print(f"CLEAN: {decoded_affiliation_string}")
    print(f"COUNTRIES: {countries}")

    for inst_id, match_strings in zip(all_mapped, all_mapped_strings):
        print(f"------{full_affiliation_dict[inst_id]['display_name']} - {inst_id}")
        if inst_id not in final_preds:
            for match_string in match_strings:
                print(f"------{match_string} ({full_affiliation_dict[inst_id]['display_name']} - {inst_id})")
                if match_string in decoded_affiliation_string:
                    print("FOUND A MATCH")
                    if not full_affiliation_dict[inst_id]['country']:
                        print("######match (no country_dict for aff ID)")
                        matched_preds.append(inst_id)
                        matched_strings.append(match_string)
                    elif not countries:
                        print("######match (no country in string)")
                        if inst_id not in multi_inst_names_ids:
                            matched_preds.append(inst_id)
                            matched_strings.append(match_string)
                        else:
                            pass
                    elif full_affiliation_dict[inst_id]['country'] in countries:
                        print("######match (country matches string)")
                        matched_preds.append(inst_id)
                        matched_strings.append(match_string)
                    else:
                        pass
                    break
                else:
                    pass
        else:
            pass
        
    # need to check for institutions that are a subset of another institution
    skip_matching = []
    for inst_id, matched_string in zip(matched_preds, matched_strings):
        for inst_id2, matched_string2 in zip(matched_preds, matched_strings):
            if (matched_string in matched_string2) & (matched_string != matched_string2):
                skip_matching.append(inst_id)
    
    if check_pred:
        for inst_id, matched_string in zip(matched_preds, matched_strings):
            for final_string in full_affiliation_dict[check_pred[0]]['final_names']:
                if matched_string in final_string:
                    skip_matching.append(inst_id)
        
    for matched_pred in matched_preds:
        if matched_pred not in skip_matching:
            final_preds.append(matched_pred)
            final_scores.append(0.95)
            final_cats.append('string_match')
            
    if (final_cats[0] == 'nothing') & (len(final_preds)>1):
        final_preds = final_preds[1:]
        final_scores = final_scores[1:]
        final_cats = final_cats[1:]
        
    # check if many names belong to same organization name (different locations)
    if (final_preds[0] != -1) & (len(final_preds)>1):
        final_display_names = [full_affiliation_dict[x]['display_name'] for x in final_preds]

        if len(final_display_names) == set(final_display_names):
            pass
        else:
            final_preds_after_removal = []
            final_scores_after_removal = []
            final_cats_after_removal = []
            preds_to_remove = get_similar_preds_to_remove(decoded_affiliation_string, final_preds)
            for temp_pred, temp_score, temp_cat in zip(final_preds, final_scores, final_cats):
                if temp_pred in preds_to_remove:
                    pass
                else:
                    final_preds_after_removal.append(temp_pred)
                    final_scores_after_removal.append(temp_score)
                    final_cats_after_removal.append(temp_cat)

            final_preds = final_preds_after_removal
            final_scores = final_scores_after_removal
            final_cats = final_cats_after_removal
            
    
    # check for multi-name institution problems (final check)
    preds_to_remove = []
    if final_preds[0] == -1:
        pass
    else:
        final_department_name_ids = [[x, str(full_affiliation_dict[x]['display_name'])] for x in final_preds if 
                       (str(full_affiliation_dict[x]['display_name']).startswith("Department of") | 
                        str(full_affiliation_dict[x]['display_name']).startswith("Department for"))]
        if final_department_name_ids:
            for temp_id in final_department_name_ids:
                if string_match_clean(temp_id[1]) not in string_match_clean(str(raw_sentence).strip()):
                    preds_to_remove.append(temp_id[0])
                elif not check_for_city_and_country_in_string(raw_sentence, countries, 
                                                              full_affiliation_dict[temp_id[0]]):
                    preds_to_remove.append(temp_id[0])
                else:
                    pass


        if any(x in final_preds for x in multi_inst_names_ids):
            # go through logic
            if len(final_preds) == 1:
                pred_name = str(full_affiliation_dict[final_preds[0]]['display_name'])
                # check if it is exact string match
                if (string_match_clean(pred_name) == string_match_clean(str(raw_sentence).strip())):
                    final_preds = [-1]
                    final_scores = [0.0]
                    final_cats = ['nothing']
                elif pred_name.startswith("Department of"):
                    if ("College" in raw_sentence) or ("University" in raw_sentence):
                        final_preds = [-1]
                        final_scores = [0.0]
                        final_cats = ['nothing']
                    elif string_match_clean(pred_name) not in string_match_clean(str(raw_sentence).strip()):
                        final_preds = [-1]
                        final_scores = [0.0]
                        final_cats = ['nothing']

            else:
                non_multi_inst_name_preds = [x for x in final_preds if x not in multi_inst_names_ids]
                if len(non_multi_inst_name_preds) > 0:
                    for temp_pred, temp_score, temp_cat in zip(final_preds, final_scores, final_cats):
                        if temp_pred not in non_multi_inst_name_preds:
                            aff_dict_temp = full_affiliation_dict[temp_pred]
                            if aff_dict_temp['display_name'].startswith("Department of"):
                                if ("College" in raw_sentence) or ("University" in raw_sentence):
                                    preds_to_remove.append(temp_pred)
                                elif (string_match_clean(str(full_affiliation_dict[temp_pred]['display_name'])) 
                                      not in string_match_clean(str(raw_sentence).strip())):
                                    preds_to_remove.append(temp_pred)
                                else:
                                    if check_for_city_and_country_in_string(raw_sentence, countries, aff_dict_temp):
                                        pass
                                    else:
                                        preds_to_remove.append(temp_pred)
                            # check for city and country
                            elif aff_dict_temp['country'] in countries:
                                pass
                            else:
                                preds_to_remove.append(temp_pred)
                else:
                    pass
        else:
            pass
    
    true_final_preds = [x for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]
    true_final_scores = [y for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]
    true_final_cats = [z for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]
    
    if not true_final_preds:
        true_final_preds = [-1]
        true_final_scores = [0.0]
        true_final_cats = ['nothing']
    return [true_final_preds, true_final_scores, true_final_cats]

def raw_data_to_predictions(df, lang_thresh, basic_thresh):
    """
    High level function to go from a raw input dataframe to the final dataframe with affiliation
    ID prediction.
    """
    # Implementing the functions above
    df['lang'] = df['affiliation_string'].apply(get_language)
    df['country_in_string'] = df['affiliation_string'].apply(get_country_in_string)
    df['comma_split_len'] = df['affiliation_string'].apply(lambda x: len([i if i else "" for i in 
                                                                          x.split(",")]))

    # Gets initial indicator of whether or not the string should go through the models
    df['affiliation_id'] = df.apply(lambda x: get_initial_pred(x.affiliation_string, x.lang, 
                                                               x.country_in_string, x.comma_split_len), axis=1)
    
    # Filter out strings that won't go through the models
    to_predict = df[df['affiliation_id']==0.0].drop_duplicates(subset=['affiliation_string']).copy()
    to_predict['affiliation_id'] = to_predict['affiliation_id'].astype('int')

    # Decode text so only ASCII characters are used
    to_predict['decoded_text'] = to_predict['affiliation_string'].apply(unidecode)

    # Get predictions and scores for each model
    to_predict['lang_pred_score'] = get_language_model_prediction(to_predict['decoded_text'].to_list(), 
                                                                  to_predict['country_in_string'].to_list())
    to_predict['basic_pred_score'] = get_basic_model_prediction(to_predict['decoded_text'].to_list(), 
                                                                to_predict['country_in_string'].to_list())

    # Get the final prediction for each affiliation string
    to_predict['affiliation_id'] = to_predict.apply(lambda x: 
                                                    get_final_prediction(x.basic_pred_score, 
                                                                         x.lang_pred_score, 
                                                                         x.country_in_string, 
                                                                         x.affiliation_string, 
                                                                         lang_thresh, basic_thresh), axis=1)

    # Merge predictions to original dataframe to get the same order as the data that was requested
    final_df = df[['affiliation_string']].merge(to_predict[['affiliation_string','affiliation_id']], 
                                                how='left', on='affiliation_string')
    
#     final_df['affiliation_id'] = final_df['affiliation_id'].fillna(-1).astype('int')
    return final_df


print("Models initialized")

# %% [markdown]
# ### Loading all gold data

# %%
def get_preds_display_names(all_preds):
    if isinstance(all_preds, float):
        return []
    elif isinstance(all_preds[0][0], int):
        if all_preds[0][0] == -1:
            return []
        else:
            return [f"{i} - {full_affiliation_dict.get(i).get('display_name')}" 
                    if full_affiliation_dict.get(i) else "-1 - None" for i in all_preds[0]]
    else:
        return []

# %%
def get_labels_display_names(all_labels):
    if isinstance(all_labels, list):
        return [f"{i} - {full_affiliation_dict.get(i).get('display_name')}" 
                    if full_affiliation_dict.get(i) else "-1 - None" for i in all_labels].copy()
    else:
        return []

# %%
def get_preds_all_or_original(all_preds, pred_type='all'):
    if isinstance(all_preds, float):
        return [-1]
    elif isinstance(all_preds[0][0], int):
        if pred_type=='all':
            return [i for i in all_preds[0]]
        else:
            final_preds = []
            
            for preds, scores, cats in zip(all_preds[0], all_preds[1], all_preds[2]):
                if cats == 'string_match':
                    pass
                else:
                    final_preds.append(preds)
            
            if not final_preds:
                final_preds = [-1]
                
            return final_preds
    else:
        return [-1]

# %% [markdown]
# Multiple different datasets were used for testing, refer to the documentation to find out more about them. The following code gathers all the datasets into a single dataframe so they can be run through the model.

# %%
multi_string = pd.read_csv(f"{institution_gold_datasets}multi_string_inst_openalex.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
multi_string['labels'] = multi_string['labels'].apply(lambda x: [int(i) for i in x.split("||||")])

cwts_1 = pd.read_csv(f"{institution_gold_datasets}cwts_related_labeled.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
cwts_1['paper_id'] = cwts_1['paper_id'].apply(lambda x: int(x[1:]))
cwts_1['labels'] = cwts_1['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])

cwts_2 = pd.read_csv(f"{institution_gold_datasets}cwts_no_relation_labeled.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
cwts_2['paper_id'] = cwts_2['paper_id'].apply(lambda x: int(x[1:]))
cwts_2['labels'] = cwts_2['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])

sampled_200 = pd.read_csv(f"{institution_gold_datasets}sampled_200_labeled.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
sampled_200['labels'] = sampled_200['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])
sampled_200['dataset'] = "gold_random"

all_gold = pd.read_csv(f"{institution_gold_datasets}gold_data_institution_parsing_labeled.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
all_gold['dataset'] = all_gold['dataset'].replace('gold_500','gold_hard').replace('gold_1000','gold_easy')
all_gold['labels'] = all_gold['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])

# %%
all_data = pd.concat([multi_string, cwts_1, cwts_2, sampled_200, all_gold], axis=0) \
    .drop_duplicates(subset=['affiliation_string'])

# %% [markdown]
# ## Testing

# %%
def get_confusion_matrix(labels, preds):
    TP=0
    FP=0
    TN=0
    FN=0
    if labels[0] == -1:
        if preds[0] != -1:
            FP = len(preds)
        else:
            TN = 1
    elif preds[0] == -1:
        FN = len(labels)
    else:
        TP = sum([1 for x in preds if x in labels])
        FP = sum([1 for x in preds if x not in labels])
        FN = sum([1 for x in labels if x not in preds])
        
    return [TP, FP, TN, FN]

# %%
def create_preview(aff_obj):
    basic_obj = aff_obj[0]
    lang_obj = aff_obj[1]
    
    basic_preds = [str(x) for x in basic_obj[0]]
    basic_scores = [round(x, 5) for x in basic_obj[1]]
    
    lang_preds = [str(x) for x in lang_obj[0]]
    lang_scores = [round(x, 5) for x in lang_obj[1]]
    
    basic_affs = [institutions.get(int(x)) for x in basic_preds]
    
    basic_ror = [x.get('ror_id') if x else "" for x in basic_affs]
    basic_aff_name = [x.get('display_name') if x else "" for x in basic_affs]
    basic_city_name = [x.get('city') if x else "" for x in basic_affs]
    basic_country_name = [x.get('country') if x else "" for x in basic_affs]
    
    lang_affs = [institutions.get(int(x)) for x in lang_preds]
    
    lang_ror = [x.get('ror_id') if x else "" for x in lang_affs]
    lang_aff_name = [x.get('display_name') if x else "" for x in lang_affs]
    lang_city_name = [x.get('city') if x else "" for x in lang_affs]
    lang_country_name = [x.get('country') if x else "" for x in lang_affs]
    
    preview_df = pd.DataFrame(zip(basic_preds, lang_preds, basic_ror, lang_ror,
                                  basic_aff_name, basic_city_name, basic_country_name,
                                  basic_scores,lang_aff_name, lang_city_name, 
                                  lang_country_name, lang_scores),
                             columns=['basic_pred','lang_pred','basic_ror','lang_ror','basic_aff_name',
                                      'basic_city_name', 'basic_country_name', 'basic_score','lang_aff_name',
                                      'lang_city_name','lang_country_name', 'lang_score'])
    
    return preview_df

# %% [markdown]
# The following code takes the gold datasets and runs all affiliation strings through the model and creates additional columns to make the results easy to explore. There is also a confusion matrix column created which can be used below to generate the precision and recall for the data. Predictions are split up between ones that are made by the model and predictions that are added on using smart string-matching.

# %%
#%%time
start_cpu_time = time.process_time()
start_wall_time = time.time()

all_preds = raw_data_to_predictions(all_data, lang_thresh=0.99, basic_thresh=0.99)\
    .merge(all_data[['paper_id','affiliation_string','labels','dataset']])

print('all_preds: -------------------------------------')
print(all_preds.head(2))

all_preds['preds_name'] = all_preds['affiliation_id'].apply(lambda x: get_preds_display_names(x))

all_preds['preds_model_and_string_matching'] = all_preds['affiliation_id'] \
    .apply(lambda x: get_preds_all_or_original(x, 'all'))
all_preds['preds_model_only'] = all_preds['affiliation_id']\
    .apply(lambda x: get_preds_all_or_original(x, 'model_only'))

all_preds['preds_model_and_string_matching'] = all_preds['preds_model_and_string_matching'] \
    .apply(lambda x: [int(i) if ~np.isnan(i) else -1 for i in x])
all_preds['preds_model_only'] = all_preds['preds_model_only']\
    .apply(lambda x: [int(i) if ~np.isnan(i) else -1 for i in x])

all_preds['labels'] = all_preds['labels'].apply(lambda x: [int(i) for i in x])
all_preds['labels_name'] = all_preds['labels'].apply(lambda x: get_labels_display_names(x))

all_preds['conf_mat_model_and_string_matching'] = all_preds\
    .apply(lambda x: get_confusion_matrix(x.labels, x.preds_model_and_string_matching), axis=1)
all_preds['conf_mat_model_only'] = all_preds.apply(lambda x: get_confusion_matrix(x.labels, 
                                                                                  x.preds_model_only), axis=1)

all_preds['has_FP'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: 1 if x[1] > 0 else 0)
all_preds['has_FN'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: 1 if x[3] > 0 else 0)

all_preds['TP'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: x[0])
all_preds['FP'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: x[1])
all_preds['TN'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: x[2])
all_preds['FN'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: x[3])

end_cpu_time = time.process_time()
end_wall_time = time.time()
print(f"CPU time used: {end_cpu_time - start_cpu_time} seconds")
print(f"Elapsed time: {(end_wall_time - start_wall_time)*1000}  μs")

print('all_preds final: -------------------------------------')
print(all_preds.head(10))

all_preds.to_csv(f'{base_save_path}all_preds.csv', index=False)

# %%


# %% [markdown]
# ### Overall Performance

# %% [markdown]
# As mentioned above, the folowing code generates the precision and recall for the data. Predictions are split up between ones that are made by the model (model_only) and predictions that are added on using smart string-matching (model_and_string_matching).

# %%
model_and_string_matching_confs_1000 = all_preds['conf_mat_model_and_string_matching'].tolist()
model_only_confs_1000 = all_preds['conf_mat_model_only'].tolist()

print("--------- MODEL WITH STRING MATCHING ---------")
print("Precision: ", round(sum([x[0] for x in model_and_string_matching_confs_1000])/
            (sum([x[0] for x in model_and_string_matching_confs_1000]) + 
             sum([x[1] for x in model_and_string_matching_confs_1000])), 3))

print("Recall: ", round(sum([x[0] for x in model_and_string_matching_confs_1000])/
            (sum([x[0] for x in model_and_string_matching_confs_1000]) + 
             sum([x[3] for x in model_and_string_matching_confs_1000])), 3))
print("")
print("--------- MODEL ONLY ---------")

print("Precision: ", round(sum([x[0] for x in model_only_confs_1000])/
            (sum([x[0] for x in model_only_confs_1000]) + sum([x[1] for x in model_only_confs_1000])), 3))

print("Recall: ", round(sum([x[0] for x in model_only_confs_1000])/
            (sum([x[0] for x in model_only_confs_1000]) + sum([x[3] for x in model_only_confs_1000])), 3))

print('FINALIZADO OK')

# %%



