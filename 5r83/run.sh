#!/bin/bash

export FG_GNINA_PATH=/home/c0065492/software/gnina
export PYTHONPATH=/home/c0065492/code/gal/
rm -r ./generated
rm ./structures/*
python rsearcher.py
