#!/bin/bash

python ../eval.py --model awe 

python ../eval.py --model bilstm 

python ../eval.py --model bilstm-max 

python ../eval.py --model lstm 
