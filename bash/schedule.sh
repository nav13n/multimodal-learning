#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh
mkdir -p data/log

python run.py trainer.max_epochs=20 experiment=experiment1 >> data/log/experiment1.txt
python run.py trainer.max_epochs=20 experiment=experiment2 >> data/log/experiment2.txt
python run.py trainer.max_epochs=20 experiment=experiment3 >> data/log/experiment3.txt
python run.py trainer.max_epochs=20 experiment=experiment4 >> data/log/experiment4.txt
python run.py trainer.max_epochs=20 experiment=experiment5 >> data/log/experiment5.txt
python run.py trainer.max_epochs=20 experiment=experiment6 >> data/log/experiment6.txt
python run.py trainer.max_epochs=20 experiment=experiment7 >> data/log/experiment7.txt
python run.py trainer.max_epochs=20 experiment=experiment8 >> data/log/experiment8.txt


