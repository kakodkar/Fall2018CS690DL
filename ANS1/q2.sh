#!/bin/bash

epoch=100
lr=1e-3
device=0
batch=60000

rm -rf q2
mkdir -p q2

python hw1_training.py -v -e ${epoch} -l ${lr} -g ${device} -n ${batch} -i torch.autograd mnist/gz

mv loss.png q2/loss.png
mv accuracy.png q2/accuracy.png
rm log
