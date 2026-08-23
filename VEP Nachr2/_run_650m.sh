#!/bin/bash
cd "C:/Users/harik/Downloads/Computational Chemistry/Variant-Effect-Predictor-for-nAChRs/VEP Nachr2" || exit 1

MODELS="logistic_regression svm_rbf random_forest lightgbm svm_linear knn gaussian_nb mlp xgboost catboost"
DF="final_mapped.xlsx"

echo "########## CLEAR CACHE ##########"
python -u -m vep_nachr2.training.runner clean

echo "########## START 3-class subunit ##########"
python -u -m vep_nachr2.training.runner compare --models $MODELS --cv-mode subunit --data-file $DF --no-remap --output-suffix _650m --n-trials 30
echo "########## DONE 3-class subunit ##########"

echo "########## START binary subunit ##########"
python -u -m vep_nachr2.training.runner compare --models $MODELS --binary --cv-mode subunit --data-file $DF --no-remap --output-suffix _650m --n-trials 30
echo "########## DONE binary subunit ##########"

echo "########## START 3-class holdout ##########"
python -u -m vep_nachr2.training.runner compare --models $MODELS --cv-mode holdout --data-file $DF --no-remap --output-suffix _650m --n-trials 30
echo "########## DONE 3-class holdout ##########"

echo "########## START binary holdout ##########"
python -u -m vep_nachr2.training.runner compare --models $MODELS --binary --cv-mode holdout --data-file $DF --no-remap --output-suffix _650m --n-trials 30
echo "########## DONE binary holdout ##########"

echo "########## ALL DONE ##########"
date
