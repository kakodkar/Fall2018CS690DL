#!/bin/bash

epoch=100
lr=1e-3
device=1

rm -rf q3
mkdir -p q3

for arch in 2 4; do
    python hw1_learning_curves.py -v -e ${epoch} -l ${lr} -g ${device} -a ${arch} -i torch.autograd mnist/gz

    mv loss.png q3/loss.${arch}.png
    mv accuracy.png q3/accuracy.${arch}.png
    rm log
    rm model.pt
done
