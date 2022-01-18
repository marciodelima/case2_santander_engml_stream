# Databricks notebook source
try: 
    import mlflow
    import mlflow.sklearn
    import pickle
    import sklearn.ensemble.forest
except:
    %pip install mlflow
    %pip install -U scikit-learn==0.21.3


# COMMAND ----------

import mlflow
import mlflow.sklearn
import pickle
import sklearn.ensemble.forest
from mlflow.tracking import MlflowClient

def registrarModelo(caminho, nome): 
    try:
        loadModel = pickle.load(open(caminho, 'rb'))
        mlflow.sklearn.log_model(loadModel, "", serialization_format="cloudpickle", registered_model_name=nome)
    except:
        pass
    
registrarModelo('/dbfs/FileStore/modelo/original_model.pkl', 'Original')
registrarModelo('/dbfs/FileStore/modelo/tunning_model.pkl', 'Tunning')

#Stage
def stagingML(nome, versao=1, stage="Production"): 
    try: 
        client = MlflowClient()
        client.transition_model_version_stage(name=nome, version=versao, stage=stage)
    except:
        pass
    
stagingML('Original')
stagingML('Tunning')


