# Databricks notebook source
# MAGIC %sql
# MAGIC create database heart

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS heart.dadosPaciente;
# MAGIC create table heart.dadosPaciente (
# MAGIC   pacienteId string,
# MAGIC   age int,
# MAGIC   sex int,
# MAGIC   chest_pain_type int,
# MAGIC   resting_blood_pressure int,
# MAGIC   cholesterol int,
# MAGIC   fasting_blood_sugar int,
# MAGIC   rest_ecg int,
# MAGIC   max_heart_rate_achieved int,
# MAGIC   exercise_induced_angina int,
# MAGIC   st_depression double,
# MAGIC   st_slope int,
# MAGIC   num_major_vessels int,
# MAGIC   thalassemia int,
# MAGIC   resultado int,
# MAGIC   probabilidade_Nao double,
# MAGIC   probabilidade_Sim double,
# MAGIC   data string
# MAGIC )
# MAGIC USING delta
# MAGIC LOCATION '/data/heart/dadosPaciente/model'

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from heart.dadosPaciente
