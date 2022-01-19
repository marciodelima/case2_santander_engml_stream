# Databricks notebook source
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
import mlflow.pyfunc
import pickle

#Modelo
model_name = 'Tunning'
model_path = f"models:/{model_name}/Production"
#logged_model = 'runs:/4cae963b5bbf4309831b329660024b1e/best model'
modelo = mlflow.sklearn.load_model(model_path)

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
    StructField("thalassemia", IntegerType(),True),
    StructField("resultado", IntegerType(),True),
    StructField("probabilidade_Nao", DoubleType(),True),
    StructField("probabilidade_Sim", DoubleType(),True),
    StructField("data", StringType(),True)
  ])

def gravarDados(dados): 
    
    dados.createOrReplaceTempView("transacao")
    dados._jdf.sparkSession().sql(""" 
        MERGE INTO heart.dadosPaciente AS tabela
        USING transacao AS dados
        ON tabela.pacienteId = dados.pacienteId
        WHEN MATCHED THEN UPDATE SET
            tabela.resultado = dados.resultado, tabela.probabilidade_Nao = dados.probabilidade_Nao, tabela.probabilidade_Sim = dados.probabilidade_Sim, tabela.data=current_date()
        WHEN NOT MATCHED THEN INSERT
            (pacienteId, age, sex, chest_pain_type, resting_blood_pressure, 
             cholesterol, fasting_blood_sugar, rest_ecg, max_heart_rate_achieved, 
             exercise_induced_angina, st_depression, st_slope, num_major_vessels, 
             thalassemia, resultado, probabilidade_Nao, probabilidade_Sim, data) VALUES (
             dados.pacienteId, dados.age, dados.sex, dados.chest_pain_type, dados.resting_blood_pressure, 
             dados.cholesterol, dados.fasting_blood_sugar, dados.rest_ecg, dados.max_heart_rate_achieved, 
             dados.exercise_induced_angina, dados.st_depression, dados.st_slope, dados.num_major_vessels, 
             dados.thalassemia, dados.resultado, dados.probabilidade_Nao, dados.probabilidade_Sim, current_date())
    """)

def prepararDados(dados): 

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

    #problemas
    #dt_transfpd = dt_transf.select("*").toPandas()
    #dt_transfpd = ps.get_dummies(dt_transfpd, drop_first=True)
    #dt_transfNew = spark.createDataFrame(dt_transfpd)
    #dt_transfNew = dt_transfpd
    
    #One Hot manual
    dt_transf = (dt_transf.withColumn("sex_male", when(col("sex") == 'male', 1).otherwise(0))
                 .withColumn("chest_pain_type_atypical angina", when(col("chest_pain_type") == 'atypical angina', 1).otherwise(0))
                 .withColumn("chest_pain_type_typical angina", when(col("chest_pain_type") == 'typical angina', 1).otherwise(0))
                 .withColumn("chest_pain_type_non-anginal pain", when(col("chest_pain_type") == 'non-anginal pain', 1).otherwise(0))
                 .withColumn("fasting_blood_sugar_lower than 120mg/ml", when(col("fasting_blood_sugar") == 'lower than 120mg/ml', 1).otherwise(0))
                 .withColumn("rest_ecg_left ventricular hypertrophy", when(col("rest_ecg") == 'left ventricular hypertrophy', 1).otherwise(0))
                 .withColumn("rest_ecg_normal", when(col("rest_ecg") == 'normal', 1).otherwise(0))
                 .withColumn("exercise_induced_angina_yes", when(col("exercise_induced_angina") == 'yes', 1).otherwise(0))
                 .withColumn("st_slope_flat", when(col("st_slope") == 'flat', 1).otherwise(0))
                 .withColumn("st_slope_upsloping", when(col("st_slope") == 'upsloping', 1).otherwise(0))
                 .withColumn("thalassemia_fixed defect", when(col("thalassemia") == 'fixed defect', 1).otherwise(0))
                 .withColumn("thalassemia_normal", when(col("thalassemia") == 'normal', 1).otherwise(0))
                 .withColumn("thalassemia_reversable defect", when(col("thalassemia") == 'reversable defect', 1).otherwise(0))
                )

    #Pegando as colunas de Entrada
    dt_transf_cols = dt_transf.select(
        'pacienteId',
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
    return dt_transf_cols.filter(col('age') > 0)

def runModelo(df_modelo): 

    #Rodando o modelo
    entrada = df_modelo.drop("pacienteId").toPandas()
    predictions_proba = modelo.predict_proba(entrada)
    predictions = modelo.predict(entrada)

    df_prob = pd.DataFrame(predictions_proba, columns = ['probabilidade_Nao','probabilidade_Sim']) 
    df_result = pd.DataFrame(predictions, columns = ['resultado'])

    df_merge = pd.concat([df_result,df_prob], axis=1, ignore_index=False)
    dados = spark.createDataFrame( pd.concat([df_modelo.select("pacienteId").toPandas(), df_merge], axis=1, ignore_index=False) )
    
    return dados

def processarDados(df_dados, epoch_id):
    if (df_dados.rdd.isEmpty()):
        return True
    #Gravar Dados
    df_dados = ( df_dados.withColumn("resultado", lit(-1))
                        .withColumn("probabilidade_Nao", lit(-1.0))
                        .withColumn("probabilidade_Sim", lit(-1.0))
                )
    dados_transf = prepararDados(df_dados)
    df_saida = runModelo(dados_transf)
    df_table = df_dados.drop("resultado","probabilidade_Nao","probabilidade_Sim").join(df_saida, ["pacienteId"])
    gravarDados(df_table)
    return True

#Streaming 
#Body vem como Binario - Parse o campo pra String
df_origem_stream = spark.readStream.format("eventhubs").options(**ehConf).load().select('body').withColumn('body', col('body').cast('string'))
#Transformando o Json
df_origem_stream=df_origem_stream.withColumn("body", from_json(col("body"),schema))
df_origem_stream=df_origem_stream.select("body.*")

df_origem_stream.writeStream.format("delta").option("checkpointLocation","{}{}".format("/capturaEventos/", "_checkpoint")).option("ignoresChanges", "true").foreachBatch(processarDados).outputMode("update").start()
