set JOB_NAME="test_job"
set BUCKET_NAME=xecg_data
set CLOUD_CONFIG=core/gcloudml-gpu.yaml
set JOB_DIR=gs://xecg_data/jobs/%JOB_NAME%
set MODULE=scripts.train_binary_classifier
set PKG_PATH=C:\Users\micha\PycharmProjects\hacktech2019\scripts
set REGION=us-central1
set RUNTIME=1.12

gcloud ml-engine jobs submit training %JOB_NAME% --package-path %PKG_PATH% --scale-tier basic --python-version 3.5 --module-name %MODULE% --job-dir %JOB_DIR% --region %REGION%