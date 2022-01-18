# Databricks notebook source
try: 
    import mlflow
    import mlflow.sklearn
    import pickle
    #import sklearn.ensemble.forest
except:
    %pip install mlflow
    #%pip install scikit-learn==0.21.3

# COMMAND ----------

# MAGIC %pip install sklearn

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pickle
import sklearn.ensemble.forest

# COMMAND ----------

#Configuracoes do EventHub
connectionString = "Endpoint=sb://case2streamnamespace.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=VT3T7LKPle7IHkFo/gZ+Ta9mI3vFEgeDpS64dT9iob0=;EntityPath=dadoshearteventhub"

ehConf = {}
ehConf["eventhubs.connectionString"] = sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(connectionString)


# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import * 
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import sklearn.ensemble.forest

#Modelo
model_name = 'Tunning'
model_path = f"models:/{model_name}/Production"
modelo = mlflow.pyfunc.load_model(model_path)

schema = StructType([ 
    StructField("pacienteId",StringType(),True), 
    StructField("age",IntegerType(),True), 
    StructField("sex",IntegerType(),True), 
    StructField("chest_pain_type", IntegerType(),True), 
    StructField("resting_blood_pressure", IntegerType(),True), 
    StructField("cholesterol", IntegerType(),True), 
    StructField("fasting_blood_sugar", IntegerType(),True), 
    StructField("rest_ecg", IntegerType(),True), 
    StructField("max_heart_rate_achieved", IntegerType(),True), 
    StructField("exercise_induced_angina", IntegerType(),True), 
    StructField("st_depression", DoubleType(),True), 
    StructField("st_slope", IntegerType(),True), 
    StructField("num_major_vessels", IntegerType(),True),
    StructField("thalassemia", IntegerType(),True)
  ])

@pandas_udf(returnType=DoubleType())
def udfModeloProb(cols): 
    entrada = pd.DataFrame.from_dict(dict(zip(cols.index, cols.values))).T
    target = modelo.predict(entrada)
    proba = modelo.predict_proba(entrada)
    return pd.Series(proba[0][target[0]])

@pandas_udf(returnType=IntegerType())
def udfModeloTarget(cols): 
    entrada = pd.DataFrame.from_dict(dict(zip(cols.index, cols.values))).T
    target = modelo.predict(entrada)
    return pd.Series(target[0])

def runModelo(dados): 

    #Inserindo valores FAKE devido ao one hot encoder / Dummies
    new_row = [{"pacienteId":"0", "age": 0, "sex": 0, "chest_pain_type": 1, "resting_blood_pressure": 145, "cholesterol": 233, "fasting_blood_sugar": 0, "rest_ecg": 0, "max_heart_rate_achieved": 150, "exercise_induced_angina": 0, "st_depression": 2.3, "st_slope": 1, "num_major_vessels": 0, "thalassemia": 0}]
    new_row1 = [{"pacienteId":"1", "age": 0, "sex": 1, "chest_pain_type": 2, "resting_blood_pressure": 145, "cholesterol": 233, "fasting_blood_sugar": 1, "rest_ecg": 1, "max_heart_rate_achieved": 150, "exercise_induced_angina": 1, "st_depression": 2.3, "st_slope": 2, "num_major_vessels": 1, "thalassemia": 1}]
    new_row2 = [{"pacienteId":"2", "age": 0, "sex": 1, "chest_pain_type": 3, "resting_blood_pressure": 145, "cholesterol": 233, "fasting_blood_sugar": 1, "rest_ecg": 2, "max_heart_rate_achieved": 150, "exercise_induced_angina": 1, "st_depression": 2.3, "st_slope": 3, "num_major_vessels": 2, "thalassemia": 2}]
    new_row3 = [{"pacienteId":"3", "age": 0, "sex": 1, "chest_pain_type": 4, "resting_blood_pressure": 145, "cholesterol": 233, "fasting_blood_sugar": 1, "rest_ecg": 2, "max_heart_rate_achieved": 150, "exercise_induced_angina": 1, "st_depression": 2.3, "st_slope": 3, "num_major_vessels": 3, "thalassemia": 3}]

    dt1 = spark.createDataFrame(new_row, schema)
    dt2 = spark.createDataFrame(new_row1, schema)
    dt3 = spark.createDataFrame(new_row2, schema)
    dt4 = spark.createDataFrame(new_row3, schema)
    dt = dados.union(dt1).union(dt2).union(dt3).union(dt4)
    
    dt_transf = (dt.withColumn("sex", when(col("sex") == 0, 'female').otherwise('male'))
            .withColumn("chest_pain_type", when(col("chest_pain_type") == 1, 'typical angina').when(col("chest_pain_type") == 2, 'atypical angina').when(col("chest_pain_type") == 3, 'non-anginal pain').otherwise('asymptomatic'))
            .withColumn("fasting_blood_sugar", when(col("fasting_blood_sugar") == 0, 'lower than 120mg/ml').otherwise('greater than 120mg/ml'))
            .withColumn("rest_ecg", when(col("rest_ecg") == 0, 'normal').when(col("rest_ecg") == 1, 'ST-T wave abnormality').otherwise('left ventricular hypertrophy'))
            .withColumn("exercise_induced_angina", when(col("exercise_induced_angina") == 0, 'no').otherwise('yes'))
            .withColumn("st_slope", when(col("st_slope") == 1, 'upsloping').when(col("st_slope") == 2, 'flat').otherwise('downsloping'))
            .withColumn("thalassemia", when(col("thalassemia") == 0, 'fix').when(col("thalassemia") == 1, 'normal').when(col("thalassemia") == 2, 'fixed defect').otherwise('reversable defect'))
         
         )
    #Tipagem
    dt_transf  = (dt_transf.withColumn('sex', col('sex').cast('string'))
                  .withColumn('chest_pain_type', col('chest_pain_type').cast('string'))
                  .withColumn('fasting_blood_sugar', col('fasting_blood_sugar').cast('string'))
                  .withColumn('rest_ecg', col('rest_ecg').cast('string'))
                  .withColumn('exercise_induced_angina', col('exercise_induced_angina').cast('string'))
                  .withColumn('st_slope', col('st_slope').cast('string'))
                  .withColumn('thalassemia', col('thalassemia').cast('string'))
                  )
    dt_transfpd = dt_transf.toPandas()
    dt_transfpd = pd.get_dummies(dt_transfpd, drop_first=True)
    dt_transf = spark.createDataFrame(dt_transfpd)
    
    entrada_modelo = struct (
        'age',
        'resting_blood_pressure',
        'cholesterol',
        'max_heart_rate_achieved',
        'st_depression',
        'num_major_vessels',
        'sex_male',
        'chest_pain_type_atypical angina',
        'chest_pain_type_non-anginal pain',
        'chest_pain_type_typical angina',
        'fasting_blood_sugar_lower than 120mg/ml',
        'rest_ecg_left ventricular hypertrophy',
        'rest_ecg_normal',
        'exercise_induced_angina_yes',
        'st_slope_flat',
        'st_slope_upsloping',
        'thalassemia_fixed defect',
        'thalassemia_normal',
        'thalassemia_reversable defect'
    )

    #Rodando o modelo
    dt_transf = dt_transf\
        .withColumn("resultado", udfModeloTarget(entrada_modelo))\
        .withColumn("probabilidade", udfModeloProb(entrada_modelo))
    dt_transf = dt_transf.select("pacienteId_1", "resultado", "probabilidade")
    dt_transf = dt_transf.withColumnRenamed("pacienteId_1", "pacienteId")
    dt = dt.join(dt_transf, ["pacienteId"])
    dt = dt.filter(col('age') > 0)
    return dt

def processarModelo(df_dados, epoch_id):
    #Transformando os dados
    df_saida = runModelo(df_dados)
    
    df_saida.write.format("delta").mode("append").save("/data/heart/dadosPaciente")


#Streaming 
#Body vem como Binario - Parse o campo pra String
df_origem_stream = spark.readStream.format("eventhubs").options(**ehConf).load().select('body').withColumn('body', col('body').cast('string'))
#Transformando o Json
df_origem_stream=df_origem_stream.withColumn("body", from_json(col("body"),schema))
df_origem_stream=df_origem_stream.select("body.*")

df_origem_stream.writeStream.format("delta").foreachBatch(processarModelo).outputMode("update").start()
#.awaitTermination()


# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import * 
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from sklearn.ensemble._forest import ForestClassifier

#Modelo
model_name = 'Tunning'
model_path = f"models:/{model_name}/Production"
modelo = mlflow.pyfunc.load_model(model_path)

print(modelo.metadata)
