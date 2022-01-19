# Databricks notebook source
# MAGIC %md
# MAGIC # Case 2 - MLFLOW - Tunning Hiper-Parametros do Modelo - LOCAL
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
import cloudpickle
import time

np.random.seed(123) #ensure reproducibility
from sklearn.model_selection import GridSearchCV

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section2'></a>

# COMMAND ----------

# MAGIC %md
# MAGIC # The Data

# COMMAND ----------

dt = pd.read_csv("../dados/heart.csv")

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

# MAGIC %md
# MAGIC # Registro do Modelo em MLFLOW
# MAGIC ## Tunning Model - Version 2 - Modelo Escolhido no HyperTunning

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = .2, random_state=10) 

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section4'></a>

# COMMAND ----------

def rodarTunning(X_train, y_train, X_test, y_test, rf_classifier):
    
    mlflow.sklearn.autolog()
    
    param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175],
                  'min_samples_split':[2,4,6,8,10],
                  'min_samples_leaf': [1, 2, 3, 4],
                  'max_depth': [5, 10, 15, 20, 25]}

    metrics = ['f1', 'recall', 'precision', 'roc_auc', 'neg_log_loss', 'neg_brier_score', 
           'average_precision', 'balanced_accuracy']
    
    grid_obj = GridSearchCV(rf_classifier,
                            return_train_score=True,
                            param_grid=param_grid,
                            scoring=metrics,
                            cv=10,
                            refit='f1')

    grid_fit = grid_obj.fit(X_train, y_train)
    rf_opt = grid_fit.best_estimator_

    mlflow.sklearn.log_model(grid_obj.best_estimator_, "best model")
    mlflow.log_metric('best score', grid_obj.best_score_)
    for k in grid_obj.best_params_.keys():
        mlflow.log_param(k, grid_obj.best_params_[k])
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    print('='*20)
    print("best params: " + str(grid_obj.best_estimator_))
    print("best params: " + str(grid_obj.best_params_))
    print('best score:', grid_obj.best_score_)
    print('='*20)
    
    print(classification_report(y_test, rf_opt.predict(X_test)))

    print('New Accuracy of Model on train set: {:.2f}'.format(rf_opt.score(X_train, y_train)*100))
    print('New Accuracy of Model on test set: {:.2f}'.format(rf_opt.score(X_test, y_test)*100))

    return rf_opt

# COMMAND ----------

rf_classifier = RandomForestClassifier(class_weight = "balanced", random_state=7)
rf_opt = rodarTunning(X_train, y_train, X_test, y_test, rf_classifier)
