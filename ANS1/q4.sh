#!/bin/bash

epoch=100
lr=1e-3
device=2
batch=60000

rm -rf q4
mkdir -p q4

python hw1_training.py -v -e ${epoch} -l ${lr} -g ${device} -n ${batch} -i my mnist/gz

mv loss.png q4/loss.png
mv accuracy.png q4/accuracy.png
rm log

for arch in 2 4; do
    python hw1_learning_curves.py -v -e ${epoch} -l ${lr} -g ${device} -a ${arch} -i my mnist/gz

    mv loss.png q4/loss.${arch}.png
    mv accuracy.png q4/accuracy.${arch}.png
    rm log
    rm model.pt
done
