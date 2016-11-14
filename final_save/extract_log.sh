#!/bin/bash

cat run.log |grep "Mean loss in" |awk '{print $NF}' > log_train_loss
cat run.log |grep "Overall VALIDATION loss" | awk '{print $NF}' > log_val_loss
cat run.log |grep "Overall VALIDATION accuracy" | awk '{print $NF}' > log_val_accu
:|paste -d',' log_train_loss - log_val_loss - log_val_accu > log_col_incorporated
