# Databricks notebook source
# MAGIC %md
# MAGIC # Case 2 - MLFLOW - Tunning Hiper-Parametros do Modelo - Databricks
# MAGIC ## Marcio de Lima

# COMMAND ----------

# MAGIC %md
# MAGIC <img style="float: left;" src="https://guardian.ng/wp-content/uploads/2016/08/Heart-diseases.jpg" width="350px"/>

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

#!pip install mlflow

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import mlflow.sklearn.utils
from sklearn.model_selection import train_test_split
import cloudpickle
import time
import pandas as pd

np.random.seed(123) #ensure reproducibility
from sklearn.model_selection import GridSearchCV

# COMMAND ----------

dt = pd.read_csv("/dbfs/FileStore/dados/heart.csv")

# COMMAND ----------

dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

# COMMAND ----------

dt['sex'][dt['sex'] == 0] = 'female'
dt['sex'][dt['sex'] == 1] = 'male'

dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'
dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'

dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'

dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 0] = 'no'
dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 1] = 'yes'

dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'
dt['st_slope'][dt['st_slope'] == 2] = 'flat'
dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'

dt['thalassemia'][dt['thalassemia'] == 1] = 'normal'
dt['thalassemia'][dt['thalassemia'] == 2] = 'fixed defect'
dt['thalassemia'][dt['thalassemia'] == 3] = 'reversable defect'

# COMMAND ----------

dt['sex'] = dt['sex'].astype('object')
dt['chest_pain_type'] = dt['chest_pain_type'].astype('object')
dt['fasting_blood_sugar'] = dt['fasting_blood_sugar'].astype('object')
dt['rest_ecg'] = dt['rest_ecg'].astype('object')
dt['exercise_induced_angina'] = dt['exercise_induced_angina'].astype('object')
dt['st_slope'] = dt['st_slope'].astype('object')
dt['thalassemia'] = dt['thalassemia'].astype('object')

# COMMAND ----------

dt = pd.get_dummies(dt, drop_first=True)

# COMMAND ----------

dt

# COMMAND ----------

# MAGIC %md
# MAGIC # Registro do Modelo em MLFLOW
# MAGIC ## Tunning Model - Version 2 - Modelo Escolhido no HyperTunning - Databricks

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = .2, random_state=10) 

# COMMAND ----------

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        #y_predict = self.model.predict(model_input)
        #y_pred_quant = self.model.predict_proba(model_input)
        return self.model.predict_proba(model_input)[:,1]
    
    def predict_proba(self, context, model_input):
        y_predict = self.model.predict(model_input)
        y_pred_quant = self.model.predict_proba(model_input)
        return y_pred_quant

# COMMAND ----------

mlflow.sklearn.autolog()

with mlflow.start_run(run_name='random_forest_model', nested=True):
    #Melhores parametros obtidos no Tunning Local
    rf_opt = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini', max_depth=10, max_features='auto',
                           max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=3,
                           min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=125, n_jobs=None, oob_score=False,
                           random_state=7, verbose=0, warm_start=False)
    rf_opt.fit(X_train, y_train)
    # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
    predictions_test = rf_opt.predict_proba(X_test)[:,1]
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_param('n_estimators', 125)
    # Use the area under the ROC curve as a metric.
    mlflow.log_metric('auc', auc_score)

    wrappedModel = SklearnModelWrapper(rf_opt)
    signature = infer_signature(X_train, wrappedModel.predict_proba(None, X_train))
    conda_env =  _mlflow_conda_env (
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)


# COMMAND ----------

from mlflow.tracking import MlflowClient

#run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

model_name = "Tunning"
model_version = "5"

mlflow.sklearn.log_model(rf_opt, "", serialization_format="cloudpickle", registered_model_name=model_name)
 
client = MlflowClient()
client.transition_model_version_stage(name=model_name, version=model_version, stage="Production")
client.transition_model_version_stage(name=model_name, version="4", stage="Staging")


# COMMAND ----------

mlflow.end_run()
