#!/bin/bash

pip uninstall autogluon-multimodal -y -q
pip uninstall autogluon-studio -y -q
pip uninstall autogluon-timeseries -y -q 
pip uninstall tensorflow -y -q
pip uninstall strands-agents-tools -y -q 
pip uninstall amazon-sagemaker-jupyter-ai-q-developer -y -q
pip uninstall -y -q sagemaker
pip uninstall -y -q sagemaker-studio
pip uninstall -y -q amazon-sagemaker-sql-magic
pip uninstall -y -q autogluon-common
pip uninstall -y -q sagemaker-core
pip uninstall -y -q dash
pip uninstall -y -q jupyter-ai
pip uninstall -y -q grpcio-status
pip uninstall -y -q opentelemetry-proto

pip install req.txt

pip install -e .

bash flow/scripts/grad_acc_flowS4.sh

