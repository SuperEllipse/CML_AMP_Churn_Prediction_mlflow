# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFR/.                                                                                                                                                                                     /INGE/MENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################
# Part 8C: Testing Deployed Model for Inferencing

# This script is used to test an API Endpoint for inferencing. This should be used after Steps 1,4,8A,8B are completed
#  The assumption is that you already have deployed the model as an API endpoint. 
# If you haven't yet, run through the initialization steps in the README file and Part 1 and Part4 and Part 8a, Part8b.
# There is one other way to use this script. You can create this as a Job pipeline and make it dependent on 8B_deploy_registered_model
# 
import os
import string
import cmlapi
from src.api import ApiUtility
from pprint import pprint

# let us try inferencing from this model  for 5 customers with features bewlo
model_Input =  {"inputs": [[0.0, 0.0, 1.0, 1.0, 58.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 20.5, 1191.4], [0.0, 0.0, 1.0, 0.0, 50.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 75.7, 3876.2], [0.0, 0.0, 1.0, 0.0, 55.0, 1.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 90.15, 4916.95], [0.0, 0.0, 0.0, 0.0, 16.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 88.45, 1422.1], [0.0, 0.0, 1.0, 0.0, 8.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 101.15, 842.9]]}

# let us get the handle to the default CML APIV2 client
client = cmlapi.default_client()
project_id = os.environ["CDSW_PROJECT_ID"]
#api_response=client.list_models(project_id=project_id, search_filter= '{\"name\":\"Churn Model Endpoint - MLOpsv1.0"}')
model_name =  os.getenv("REGISTERED_MODEL_NAME" ,  "Churn Model Endpoint - MLOpsv1.0")
api_response=client.list_models(project_id=project_id,     search_filter=f'{{"name":"{model_name}"}}')

model_id = api_response.models[0].id
access_key = api_response.models[0].access_key
print(model_id,  project_id, access_key) #model id\


from  cml import models_v1
# let us try inferencing from this model for 5 Customer with feature values below. Note that each list provides the feature values for one customer
model_Input =  {"inputs": [[0.0, 0.0, 1.0, 1.0, 58.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 20.5, 1191.4], [0.0, 0.0, 1.0, 0.0, 50.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 75.7, 3876.2], [0.0, 0.0, 1.0, 0.0, 55.0, 1.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 90.15, 4916.95], [0.0, 0.0, 0.0, 0.0, 16.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 88.45, 1422.1], [0.0, 0.0, 1.0, 0.0, 8.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 101.15, 842.9]]}
api_response = models_v1.call_model(model_access_key=access_key,ipt=model_Input)
# Let us fetch the predictions
# print(api_response['response'])
pprint(api_response)