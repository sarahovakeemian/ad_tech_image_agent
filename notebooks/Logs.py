# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Initialize the Databricks SDK
w = WorkspaceClient()

# COMMAND ----------

# Get the logs for a served model
logs = w.serving_endpoints.logs(name="pet_ad_image_gen", served_model_name="pet_ad_image_gen")

print(logs)
