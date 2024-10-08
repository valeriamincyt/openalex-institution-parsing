# %%
import pickle
import boto3
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
#from pyspark.sql import SparkSession
#sc = spark.sparkContext
#from pyspark.sql import SQLContext
#from pyspark.sql import functions as F
#from pyspark.sql.window import Window
#from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType, DoubleType, StructType, StructField
#sqlContext = SQLContext(sc)

#Vale
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
#sc = spark.sparkContext         # Since Spark 2.0 'spark' is a SparkSession object that is by default created upfront and available in Spark shell, you need to explicitly create SparkSession object by using builder
spark = SparkSession.builder.getOrCreate()
#sc = SparkContext().getOrCreate()
sc = SparkContext._active_spark_context #devuelve la instancia existente
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType, DoubleType, StructType, StructField
sqlContext = SQLContext(sc,spark)
# %%
# These paths should be changed to wherever you want to save the general data and where you want to save
# iteration specific data
base_save_path = "./"
iteration_save_path = "./institutional_affiliation_classification/"
rutaDatos = "../Datos/"
os.mkdir("./institutional_affiliation_classification/")

# %% [markdown]
# ### Getting all data (From saved OpenAlex DB snapshot)

# %%
institutions = spark.read.parquet(f"{rutaDatos}OA_static_institutions_single_file.parquet") \
    .filter(F.col('ror_id')!='')

institutions['affiliation_id'] = institutions['affiliation_id'].apply(lambda x: x.split("/")[-1])

# %%
print('institutions.cache().count() --------------------------------------')
print(institutions.cache().count())

# %%
df_affil = pd.read_csv(f"{rutaDatos}Insumos_M1/static_affiliations_santi.csv")
#df_affil = df_affil.rename(columns={'affiliation_ids': 'affiliation_id'})
df_affil.to_parquet(f"{rutaDatos}static_affiliations.parquet")
print('Se crea el archivo static_affiliations.parquet con los datos de Insumos_M1/static_affiliations_santi.csv')

# %%
affiliations = spark.read.parquet(f"{rutaDatos}static_affiliations.parquet")
affiliations['affiliation_id'] = affiliations['affiliation_id'].apply(lambda x: x.split("/")[-1])

# %%
print('affiliations.cache().count() --------------------------------------')
print(affiliations.cache().count())
print(affiliations.head(1))

# %% [markdown]
# #### Getting ROR aff strings

# %%
dedup_affs = affiliations.select(F.trim(F.col('original_affiliation')).alias('original_affiliation'), 'affiliation_id')\
.filter(F.col('original_affiliation').isNotNull())\
.filter(F.col('original_affiliation')!='')\
.withColumn('aff_len', F.length(F.col('original_affiliation')))\
.filter(F.col('aff_len')>2)\
.groupby(['original_affiliation','affiliation_id']) \
.agg(F.count(F.col('affiliation_id')).alias('aff_string_counts'))

# %%
dedup_affs.cache().count()

# %%
ror_data = spark.read.parquet(f"{rutaDatos}ror_strings.parquet") \
.select('original_affiliation','affiliation_id')

# %%
ror_data.cache().count()

# %% [markdown]
# ### Gathering training data
# 
# Since we are looking at all institutions, we need to up-sample the institutions that don't have many affiliation strings and down-sample the institutions that have large numbers of strings. There was a balance here that needed to be acheived. The more samples that are taken for each institution, the more overall training data we will have and the longer our model will take to train. However, more samples also means more ways of an institution showing up in an affiliation string. The number of samples was set to 50 as it was determined this was a good optimization point based on affiliation string count distribution and time it would take to train the model. However, unlike in V1 where we tried to keep all institutions at 50, for V2 we gave additional samples for institutions with more strings available. Specifically, we allowed those institutions to have up to 25 additional strings, for a total of 75.

# %%
num_samples_to_get = 50 

# %%
w1 = Window.partitionBy('affiliation_id')

filled_affiliations = dedup_affs \
    .join(ror_data.select('affiliation_id'), how='inner', on='affiliation_id') \
    .select('original_affiliation','affiliation_id') \
    .union(ror_data.select('original_affiliation','affiliation_id')) \
    .filter(~F.col('affiliation_id').isNull()) \
    .dropDuplicates() \
    .withColumn('random_prob', F.rand(seed=20)) \
    .withColumn('id_count', F.count(F.col('affiliation_id')).over(w1)) \
    .withColumn('scaled_count', F.lit(1)-((F.col('id_count') - F.lit(num_samples_to_get))/(F.lit(3500000) - F.lit(num_samples_to_get)))) \
    .withColumn('final_prob', F.col('random_prob')*F.col('scaled_count'))

# %%
filled_affiliations.select('affiliation_id').distinct().count()

# %%
less_than = filled_affiliations.dropDuplicates(subset=['affiliation_id']).filter(F.col('id_count') < num_samples_to_get).toPandas()
less_than.shape

# %%
less_than.sample(10)

# %%
temp_df_list = []
for aff_id in less_than['affiliation_id'].unique():
    temp_df = less_than[less_than['affiliation_id']==aff_id].copy()
    help_df = temp_df.sample(num_samples_to_get - temp_df.shape[0], replace=True)
    temp_df_list.append(pd.concat([temp_df, help_df], axis=0))
less_than_df = pd.concat(temp_df_list, axis=0)

# %%
less_than_df.shape

# %%
# only install fsspec and s3fs
less_than_df[['original_affiliation', 'affiliation_id']].to_parquet(f"{iteration_save_path}lower_than_{num_samples_to_get}.parquet")

# %%
w1 = Window.partitionBy('affiliation_id').orderBy('random_prob')

more_than = filled_affiliations.filter(F.col('id_count') >= num_samples_to_get) \
.withColumn('row_number', F.row_number().over(w1)) \
.filter(F.col('row_number') <= num_samples_to_get+25)

# %%
more_than.cache().count()

# %%
more_than.select('original_affiliation', 'affiliation_id') \
.coalesce(1).write.mode('overwrite').parquet(f"{iteration_save_path}more_than_{num_samples_to_get}")

# %%


# %%



