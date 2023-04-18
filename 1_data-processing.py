#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('jt -f consolamono -fs 14')


# # Import Libraries

# In[142]:


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
import pyspark.sql.functions as func
from pyspark.sql.functions import col

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ## Spark Session

# In[143]:


spark =  SparkSession.builder.master("local").appName("hive_stackoverflow").enableHiveSupport().getOrCreate()


# In[144]:


# Switch to stackoverflow_dataset
spark.sql("USE stackoverflow;")

# Allow ACID transaction on table 
spark.sql("""set hive.support.concurrency = true;""")
spark.sql("""set hive.txn.manager = org.apache.hadoop.hive.ql.lockmgr.DbTxnManager;""")


# In[145]:


# Display Tables in the selected databases
spark.sql("SHOW TABLES;").show()


# In[ ]:


# Insert data into questions from questions_so
# spark.sql("""INSERT OVERWRITE TABLE questions SELECT * FROM questions_so;""")


# In[ ]:


questions_data = spark.sql("""SELECT * FROM questions;""")


# In[ ]:


# Remove Unwanted Columns
questions_data = questions_data.drop('OwnerUserId')
questions_data = questions_data.drop('ClosedDate')
questions_data = questions_data.drop('Body')


# In[ ]:


print((questions_data.count(), len(questions_data.columns)))


# In[ ]:


questions_data.printSchema()


# In[ ]:


questions_data.show(5)


# # Data Cleaning and Processing

# In[ ]:


# Delete first row
questions_data = questions_data.where(questions_data.title!="Title")

# Convert to Lowercase
# spark.sql("""UPDATE questions SET title = LOWER(title);""")
questions_data_cleaned = questions_data.select("*", func.lower(func.col("title")).alias("title1"))
questions_data_cleaned.limit(10).toPandas()


# In[ ]:


# Remove Punctuations from title
# spark.sql("""UPDATE questions SET title = REGEXP_REPLACE(title, "[^a-zA-Z0-9 \#+.]", "");""")
questions_data_cleaned = questions_data_cleaned.select("*", func.regexp_replace("title1", "[^a-zA-Z0-9 \#+.]", "").alias("title2"))
questions_data_cleaned.limit(10).toPandas()


# In[ ]:


# Remove Stopwords
# stopword_regex = "i|me|my|myself|we|our|ours|ourselves|you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves|what|which|who|whom|this|that|these|those|am|is|are|was|were|be|been|being|have|has|had|having|do|does|did|doing|a|an|the|and|but|if|or|because|as|until|while|of|at|by|for|with|about|against|between|into|through|during|before|after|above|below|to|from|up|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|s|t|can|will|just|don|should|now"

# questions_data_cleaned = questions_data_cleaned.select("*", func.regexp_replace("title2", stopword_regex, "").alias("title3"))
# questions_data_cleaned.limit(10).toPandas()


# In[ ]:


# Delete Created Columns
questions_data_cleaned = questions_data_cleaned.drop('title')
questions_data_cleaned = questions_data_cleaned.drop('title1')
# questions_data_cleaned = questions_data_cleaned.drop('title3')


# In[ ]:


questions_data_cleaned.limit(10).toPandas()


# In[ ]:


# Rename column
final_cleaned_data = questions_data_cleaned.withColumnRenamed("title2","title")

# Print Schema
final_cleaned_data.printSchema()

# Display top 10 rows
final_cleaned_data.limit(10).toPandas()


# In[ ]:





# In[ ]:


final_cleaned_data = final_cleaned_data.toPandas()


# In[140]:


final_cleaned_data.head()


# In[162]:


final_cleaned_data[final_cleaned_data['title']=='']


# In[141]:


final_cleaned_data.to_csv("/home/aditya_bagad2/cleaned_questions.csv", index=False, header=True, sep=',')


# ## Clean Tags Data

# In[151]:


tags_data = spark.sql("""SELECT * FROM tags_so;""")


# In[153]:


tags_data.show(5)


# In[155]:


# Delete first row
tags_data = tags_data.where(tags_data.tag!="Tag")


# In[156]:


tags_data = tags_data.toPandas()


# In[157]:


tags_data.head()


# In[158]:


tags_data.to_csv("/home/aditya_bagad2/tags.csv", index=False, header=True, sep=',')


# In[ ]:




