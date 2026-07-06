#!/bin/bash
export AIRFLOW_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/airflow"
echo "AIRFLOW_HOME défini sur : $AIRFLOW_HOME"