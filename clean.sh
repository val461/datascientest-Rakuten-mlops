#!/bin/sh
docker compose down
rm -vi data/preprocessed/*
rm -vi mlruns/*
rm -vi models/*
docker volume rm rakuten-mlops_airflow_home rakuten-mlops_grafana-storage
