#!/bin/bash
python ../eval.py --model awe --prototype

python ../eval.py --model bilstm --prototype

python ../eval.py --model bilstm-max --prototype

python ../eval.py --model lstm --prototype
