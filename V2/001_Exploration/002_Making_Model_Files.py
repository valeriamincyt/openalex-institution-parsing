#002_Making_Model_Files
#!pip install pip --upgrade
#!pip install unidecode
#!pip install langdetect

# %%
import pickle
import json
#import redshift_connector
import pandas as pd
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
import numpy as np
import re
import os

from unidecode import unidecode
from collections import Counter
from math import ceil
from langdetect import detect
from random import sample

# %% [markdown]
# ## Support Files

# %% [markdown]
# Throughout the modeling process, some of the model artifacts needed to be updated and so this notebook was used to quickly update those files.

# %%
# location where current files are located
base_save_path = "./"
rutaDatos ="../Datos/"
curr_model_artifacts_location = f"{rutaDatos}/institution_tagger_v2_artifacts/"

# Load the needed files
with open(f"{curr_model_artifacts_location}departments_list.pkl", "rb") as f:
    departments_list = pickle.load(f)

print("Loaded list of departments")

with open(f"{curr_model_artifacts_location}countries_list_flat.pkl", "rb") as f:
    countries_list_flat = pickle.load(f)

print("Loaded flat list of countries")

with open(f"{curr_model_artifacts_location}countries.json", "r") as f:
    countries_dict = json.load(f)

print("Loaded countries dictionary")

with open(f"{curr_model_artifacts_location}city_country_list.pkl", "rb") as f:
    city_country_list = pickle.load(f)

print("Loaded strings of city/country combinations")

# %%


# %%


# %% [markdown]
# ### Looking at ROR

# %%
def get_geoname_admin(address_dict):
    try:
        geoname_admin = address_dict['geonames_city']['geonames_admin1']['name']
    except:
        geoname_admin = "None"
        
    return geoname_admin

# %%
def get_final_region(ror_state, ror_region):
    if isinstance(ror_state, str):
        return ror_state
    elif isinstance(ror_region, str):
        return ror_region
    else:
        return None

# %%
def check_for_backwards_univ(curr_names):
    names = curr_names.copy()
    for one_name in curr_names:
        split_name = one_name.split(" ")
        if len(split_name) == 3:
            if (split_name[0] == 'University') & (split_name[1] == 'of'):
                names.append(f"{split_name[2]} University")
        elif len(split_name) == 2:
            if (split_name[1] == 'University'):
                names.append(f"University of {split_name[0]}")
        else:
            pass
    return names

# %%
def add_names_to_list(all_names):
    names = all_names.copy()
    if "Harvard University" in names:
        names.append("Harvard Medical School")
    elif "University of Oxford" in names:
        names.append("Oxford University")
    else:
        pass
    
    return names

# %%
def get_exact_names(name, aliases, acronyms, labels):
    all_names = [name] + aliases + acronyms + [i['label'] for i in labels]
    all_names = add_names_to_list(all_names)
    all_names = [x for x in all_names if ~x.startswith('Department of')]
    all_names_clean = [string_match_clean(x) for x in all_names]
    return [x for x in all_names_clean if len(x) > 4]

# %%
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
    
    # take out spaces, commas, dashes, periods, etcs
    text = re.sub("[^0-9a-zA-Z]", "", text)
    
    return text

# %%
def list_of_all_names(oa_name, ror_names, extra_names, use_extra_names=False):
    banned_names = ['UniversityHospital','Coastal','Brunswick','Continental']
    if isinstance(ror_names, list):
        pass
    else:
        ror_names = []
        
    if isinstance(oa_name, str):
        oa_string = [string_match_clean(oa_name)]
    else:
        oa_string = []
        
    if use_extra_names:
        if isinstance(extra_names, list):
            pass
        else:
            extra_names = []
    else:
        extra_names = []
    
    return [x for x in list(set(oa_string+ror_names+extra_names)) if 
            ((len(x) > 4) & (x not in banned_names))]

# %%
# this file is not provided but the needed data is all institutions in OpenAlex
# with the following columns: 'ror_id','affiliation_id','display_name'
institutions_df = pd.read_parquet(f"{rutaDatos}OA_static_institutions_single_file.parquet")
institutions_df['ror_id'] = institutions_df['ror_id'].apply(lambda x: x.split("/")[-1])
institutions_df['affiliation_id'] = institutions_df['affiliation_id'].apply(lambda x: x.split("/")[-1])
# %%
# institutions = institutions_df.set_index('affiliation_id').to_dict(orient='index')

# %%
ror = pd.read_json(f"{rutaDatos}v1.55-2024-10-31-ror-data.json")

# %%
ror['address'] = ror['addresses'].apply(lambda x: x[0])
ror['ror_id'] = ror['id'].apply(lambda x: x.split("/")[-1])
#ror['types'] = ror['types'].apply(lambda x: x[0])
ror['types'] = ror['types'].apply(lambda x: x[0] if len(x)>0 else "")

# %%
ror['country_code'] = ror['country'].apply(lambda x: x['country_code'])
ror['country_name'] = ror['country'].apply(lambda x: x['country_name'])
ror['city'] = ror['address'].apply(lambda x: x['city'])
ror['state'] = ror['address'].apply(lambda x: x['state'])
ror['region'] = ror['address'].apply(get_geoname_admin)
ror['ror_id'] = ror['id'].apply(lambda x: x.split("/")[-1])
# %%
ror_to_join = ror[['ror_id','name','status','types','aliases','acronyms','labels','city',
                   'state','region','country_name']].copy()

ror_to_join.columns = ['ror_id','name','status','types','aliases','acronyms','labels','city',
                       'temp_state','temp_region','country']

# %%
ror_to_join['state'] = ror_to_join.apply(lambda x: get_final_region(x.temp_state, x.temp_region), axis=1)

# %%
inst_ror = ror_to_join.merge(institutions_df[['ror_id','affiliation_id','display_name']], 
                             how='inner', on='ror_id')
inst_ror.shape

# %% [markdown]
# #### Getting file of multi-institution names

# %%
multi_inst_names_df = ror_to_join[['ror_id','name']].merge(institutions_df[['ror_id','affiliation_id']], 
                                                        how='left', on='ror_id') \
['name'].value_counts().reset_index()

# %%
print('multi_inst_names_df.head(10) ----------------------------------------------------------')
print(multi_inst_names_df.head(10))

# %%
multi_inst_names = multi_inst_names_df[multi_inst_names_df['count']>1].iloc[:,0].tolist()

# %%
multi_inst_names_ids = inst_ror[inst_ror['name'].isin(multi_inst_names)]['affiliation_id'].tolist()

# %%
with open(f"{curr_model_artifacts_location}multi_inst_names_ids.pkl", "wb") as f:
    pickle.dump(multi_inst_names_ids, f)

# %% [markdown]
# ### Getting Mapping of Inactive Institutions

# %% [markdown]
# There are institutions in ROR that are listed as "Withdrawn" or "Inactive". There was some thought to use the old data associated with these ROR IDs and apply them to successors but for this model, we decided to hold off on doing this because we were unsure if there would be a benefit to doing so. Therefore, the code is provided but this data was not used in building the model.

# %%
def get_successors_from_relationships(relationships):
    successors = []
    parents = []
    for relationship in relationships:
        if relationship['type'] == 'Successor':
            successors.append(relationship['id'].split("/")[-1])
        elif relationship['type'] == 'Parent':
            parents.append(relationship['id'].split("/")[-1])
        else:
            pass
    return [successors, parents]

# %%
def get_extra_names(ror_id):
    if ror_id in successor_dict.keys():
        extra_names = []
        for old_id in successor_dict[ror_id]['ror_id']:
            extra_names += old_name_data[old_id]['successor_names']
        
        extra_names = list(set(extra_names))
    else:
        extra_names = []
        
    return extra_names

# %%
withdrawn_or_inactive_df = ror[ror['status'].isin(['withdrawn','inactive'])].copy()
withdrawn_or_inactive_df.shape

# %%
withdrawn_or_inactive_df['successors_parents'] = withdrawn_or_inactive_df['relationships'] \
    .apply(get_successors_from_relationships)

# %%
withdrawn_or_inactive_df['successors'] = withdrawn_or_inactive_df['successors_parents'].apply(lambda x: x[0])
withdrawn_or_inactive_df['parents'] = withdrawn_or_inactive_df['successors_parents'].apply(lambda x: x[1])

# %%
withdrawn_or_inactive_df['successor_len'] = withdrawn_or_inactive_df['successors'].apply(len)

# %%
to_add_to_successors = withdrawn_or_inactive_df[withdrawn_or_inactive_df['successor_len']==1].copy()
to_add_to_successors['successor'] = to_add_to_successors['successors'].apply(lambda x: x[0])
to_add_to_successors.shape

# %%
to_add_to_successors['successor_names'] = to_add_to_successors.apply(lambda x: get_exact_names(x['name'], 
                                                                                               x.aliases, 
                                                                                               x.acronyms, 
                                                                                               x.labels), 
                                                                     axis=1)

# %%
old_name_data = to_add_to_successors.set_index('ror_id')[['successor_names']].to_dict(orient='index')

# %%
successor_dict = to_add_to_successors.groupby('successor')['ror_id'].apply(list).reset_index()\
    .set_index('successor').to_dict(orient='index')

# %% [markdown]
# ### Getting ROR String Matching File and Affiliation Dictionary

# %%
inst_ror['extra_names'] = inst_ror['ror_id'].apply(get_extra_names)

# %%
inst_ror['exact_names'] = inst_ror.apply(lambda x: get_exact_names(x['name'], x.aliases, x.acronyms, x.labels), axis=1)

# %%
inst_ror['final_names'] = inst_ror.apply(lambda x: list_of_all_names(x.display_name, x.exact_names, x.extra_names, use_extra_names=False), axis=1)

# %%
new_affiliation_dict = inst_ror.set_index('affiliation_id')[['display_name','city','state',
                                                             'country','final_names','ror_id','types']] \
.to_dict(orient='index')

# %%
with open(f"{curr_model_artifacts_location}full_affiliation_dict.pkl", "wb") as f:
    pickle.dump(new_affiliation_dict, f)

# %%


# %% [markdown]
# ### Updating the city/country file

# %% [markdown]
# This file is used to check the affiliation string to make sure it doesn't exactly match up with a city/region/country combo with no additional information.

# %%
city_region_country = inst_ror.drop_duplicates(subset=['city','country']).copy()
city_region_country.shape

# %%
new_city_country_list = list(set([f"{i}{j}" for i,j in zip(city_region_country['city'].tolist(), 
                                   city_region_country['country'].tolist())] + 
         [f"{i}{j}{k}"for i,j,k in zip(city_region_country['city'].tolist(), 
                                             city_region_country['state'].tolist(),
                                             city_region_country['country'].tolist()) if j ] + 
         [f"{i}{j}" for i,j in zip(city_region_country['state'].tolist(), 
                                   city_region_country['country'].tolist()) if i] + 
         [f"{i}" for i in city_region_country['country'].tolist()] + 
         [f"{i}" for i in city_region_country['state'].tolist() if i]))

new_city_country_list = list(set([string_match_clean(x) for x in new_city_country_list]))

# %%
len(new_city_country_list)

# %%
with open(f"{curr_model_artifacts_location}city_country_list.pkl", "wb") as f:
    pickle.dump(new_city_country_list, f)

# %% [markdown]
# ### Flat country file is up to date

# %% [markdown]
# Flat country file is needed to search for country in the string for the model.

# %%
len(list(set(countries_list_flat)))

# %%
all_countries = []
for i in countries_dict.values():
    all_countries += i

# %%
len(list(set(all_countries)))

# %%
with open(f"{curr_model_artifacts_location}countries_list_flat.pkl", "wb") as f:
    pickle.dump(list(set(all_countries)), f)

# %% [markdown]
# ### Departments list update

# %% [markdown]
# Takes the old department list and updates it with additional department names.

# %%
with open(f"{base_save_path}ror_string_beginnings/Education_dept", 'r') as f:
    education_dept_begs = f.readlines()

education_dept_begs = list(set([x.rstrip('\n') for x in education_dept_begs]))

# %%
departments_list = ['Psychology','Nephrology','Other departments','Other Departments','Nursing & Midwifery',
                    'Literature and Creative Writing','Neuroscience','Engineering','Computer Science',
                    'Chemistry','Biology','Medicine']

# %%
new_departments_list = list(set(departments_list + education_dept_begs))

# %%
with open(f"{curr_model_artifacts_location}departments_list.pkl", "wb") as f:
    pickle.dump(new_departments_list, f)

# %%


# %% [markdown]
# ### Make affiliation IDs integers

# %%
with open(f"{curr_model_artifacts_location}affiliation_vocab.pkl", "rb") as f:
    affiliation_vocab_basic = pickle.load(f)
    
new_affiliation_vocab_basic = {int(i):int(j) for j,i in affiliation_vocab_basic.items()}

print("Loaded basic affiliation vocab")

# %%
with open(f"{curr_model_artifacts_location}affiliation_vocab.pkl", "wb") as f:
    pickle.dump(new_affiliation_vocab_basic, f)

# %%

print('FINALIZADO OK')

