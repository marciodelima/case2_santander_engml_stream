# Databricks notebook source
#Configuracoes do EventHub
connectionString = "Endpoint=sb://case2streamnamespace.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=VT3T7LKPle7IHkFo/gZ+Ta9mI3vFEgeDpS64dT9iob0=;EntityPath=dadoshearteventhub"

ehConf = {}
ehConf["eventhubs.connectionString"] = sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(connectionString)


# COMMAND ----------

import os

from pyspark.sql.functions import * 
from pyspark.sql import Row
from pyspark.sql.types import * 
from random import randint
import uuid

id = uuid.uuid4()
dados = [
    {
        'pacienteId': str(id),
        'age': randint(18,99),
        'sex': randint(0,1),
        'chest_pain_type': randint(1,4),
        'resting_blood_pressure': randint(0,1),
        'cholesterol': randint(120,400),
        'fasting_blood_sugar': randint(0,1),
        'rest_ecg': randint(0,2),
        'max_heart_rate_achieved': randint(80,200),
        'exercise_induced_angina': randint(0,1),
        'st_depression': 2.3,
        'st_slope': randint(1,3),
        'num_major_vessels': randint(0,3),
        'thalassemia': randint(1,3)
    }
]


df = spark.createDataFrame(dados)
display(df)

# COMMAND ----------

df_json = (df.select(to_json(struct(col("*"))).alias("body")))
display(df_json)

# COMMAND ----------

#Envio ao EventHub
df_json.write.format("eventhubs").options(**ehConf).save()
