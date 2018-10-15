Homework 1
===

- Jianfei Gao

- 2018/08/27

# Q1

## 1

If we only use linear activations in a neural network, it will just be a linear system which means that output will just be a linear combination of input. This will limit the power of neural network.

## 2

ReLU is a non-linear activation, so it can help to make output not just be a linear combination of input. In this aspect, it solves the problem.

## 3

When ReLU activation gets an input without any positive value. It will get all zero gradients, which causes all the layers after it to stop learning.

## 4

Suppose a supervised task.
- Given $\{(x_i^\text{tr}, y_i^\text{tr})\}_{i = 1}^{n^\text{tr}}$, $(x_i^\text{tr}, y_i^\text{tr}) \sim \text{P}(x, y)$
- Given $\{(x_i^\text{te}, y_i^\text{te})\}_{i = 1}^{n^\text{te}}$, $(x_i^\text{te}, y_i^\text{te}) \sim \text{P}(x, y)$
- Model $\text{P}(y|x)$

It can be converted into two unsupervised tasks

1. $\text{P}(x, y)$
   - Given $\{(x_i^\text{tr}, y_i^\text{tr})\}_{i = 1}^{n^\text{tr}}$, $(x_i^\text{tr}, y_i^\text{tr}) \sim \text{P}(x, y)$
   - Given $\{(x_i^\text{te}, y_i^\text{te})\}_{i = 1}^{n^\text{te}}$, $(x_i^\text{te}, y_i^\text{te}) \sim \text{P}(x, y)$
   - Model $\text{P}(x, y)$

2. $\text{P}(x)$
   - Given $\{x_i^\text{tr}\}_{i = 1}^{n^\text{tr}}$, $x_i^\text{tr} \sim \text{P}(x) = \int_y \text{P}(x, y) dy$
   - Given $\{x_i^\text{te}\}_{i = 1}^{n^\text{te}}$, $x_i^\text{te} \sim \text{P}(x) = \int_y \text{P}(x, y) dy$
   - Model $\text{P}(x)$

Then $\text{P}(y|x) = \frac{\text{P}(x, y)}{\text{P}(x)}$.

## 5

In multitask learning, we are learning all labels jointly
$$
\text{P}(y_1, y_2, \cdots, y_T | x),
$$
but in transfer learning, we are using other labels help to learn a label, e.g.
$$
\text{P}(y_1 | x, y_2, y_3, \cdots, y_T).
$$

We should use transfer learning to train. First, we train an image feature extractor and a classifier used to distinguish dogs and cats over extracted features jointly on the large dataset. Then, we copy the model and parameters of image feature extractor, use it to extract features of facial images, and train another classifier used to distinguish male and female over extracted features separately.

# Q2

**Random seed of PyTorch is fixed by 29 for all the following codes**.

## 1

I run command

```bash
python hw1_training.py -e 100 -l 0.001 -i torch.autograd -g 0 -v mnist/gz
```

to train samples from data folder `mnist/gz` by **AutogradNeuralNetwork** with learning rate 0.001 for 100 epochs on GPU-0.

## 2

There is an obvious gap between performance of training and testing data after convergence.

- Loss VS. Epochs

  ![loss](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q2\loss.png)

- Accuracy VS. Epochs

  ![accuracy](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q2\accuracy.png)

# Q3

**I still use the same code provided in original files to plot, so the label of x-axis in this question indeed means number of training samples **.

I run command

```bash
python hw1_learning_curves.py -e 100 -l 0.001 -i torch.autograd -g 0 -a 2 -v mnist/gz
python hw1_learning_curves.py -e 100 -l 0.001 -i torch.autograd -g 0 -a 4 -v mnist/gz
```

to train samples from data folder `mnist/gz` by **BasicNeuralNetwork** with learning rate 0.001 for 100 epochs on GPU-0 for shape (784, 10) and (784, 300, 100, 10).

For early stopping, I will only keep the model with best test accuracy in memory. If test accuracy become worse, it will load the best model from disk, otherwise, it will save to disk.

Here are learning curves

- (784, 10)

   - Loss VS. Epochs

     ![loss.2](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q3\loss.2.png)

   - Accuracy VS. Epochs

     ![accuracy.2](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q3\accuracy.2.png)

- (784, 300, 100, 10)

   - Loss VS. Epochs

     ![loss.4](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q3\loss.4.png)

   - Accuracy VS. Epochs

     ![accuracy.4](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q3\accuracy.4.png)

With number of training samples increasing, the gap between performance of training and testing data are becoming smaller and smaller. The shallower neural network (784, 10) has a smaller gap at starting point than deeper neural network (784, 300, 100, 10).

# Q4

## 1

See uploaded files.

## 2

- Suppose input $x$ of a layer is a 2D matrix of $\text{#Features}_1 \times \text{#Samples}$
- Suppose output $y$ of a layer is a 2D matrix of $\text{#Features}_2 \times \text{#Samples}$
- Suppose possible one-hot target $y^{\{0,1\}}$ is a 2D matrix of $\text{#Classes} \times \text{#Samples}$

- **Cross Entropy Loss and Softmax Activation**

   For feed forward process
   $$
   L = \text{cross_entropy}(\text{softmax}(x), y^{\{0,1\}})
   $$
   The back propagate process will be
   $$
   \frac{\partial L}{\partial x} = \frac{1}{\text{#Samples}} (\text{softmax}(x) - y^{\{0,1\}})
   $$

- **Linear**

   For feed forward process
   $$
   y = Wx + b
   $$
   The back propagate process with gradient $\frac{\partial L}{\partial y}$ of $y$ will be
   $$
   \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \\
   \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} x^\text{T} \\
   \frac{\partial L}{\partial x} = W^\text{T} \frac{\partial L}{\partial y} \\
   $$

- **ReLU Activation**

   For feed forward process
   $$
   y = \text{relu}(x)
   $$
   The back propagate process with gradient $\frac{\partial L}{\partial y}$ of $y$ will be
   $$
   \frac{\partial L}{\partial x_{ij}} = \begin{cases}
      0 & x_{ij} < 0 \\
      \frac{\partial L}{\partial y_{ij}} & \text{o.w.}
   \end{cases}
   $$








## 3

I run command

```bash
python hw1_training.py -e 100 -l 0.001 -i my -g 0 -v mnist/gz
```

to train samples from data folder `mnist/gz` by **BasicNeuralNetwork** with learning rate 0.001 for 100 epochs on GPU-0.

- Loss VS. Epochs

  ![loss](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q4\loss.png)


- Accuracy VS. Epochs

  ![accuracy](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q4\accuracy.png)

## 4

I run command

```bash
python hw1_learning_curves.py -e 100 -l 0.001 -i my -g 0 -a 2 -v mnist/gz
python hw1_learning_curves.py -e 100 -l 0.001 -i my -g 0 -a 4 -v mnist/gz
```

to train samples from data folder `mnist/gz` by **BasicNeuralNetwork** with learning rate 0.001 for 100 epochs on GPU-0 for shape (784, 10) and (784, 300, 100, 10).

For early stopping, I will only keep the model with best test accuracy in memory. If test accuracy become worse, it will load the best model from disk, otherwise, it will save to disk.

Here are learning curves

- (784, 10)

   - Loss VS. Epochs

     ![loss.2](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q4\loss.2.png)

   - Accuracy VS. Epochs

     ![accuracy.2](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q4\accuracy.2.png)

- (784, 300, 100, 10)

   - Loss VS. Epochs

     ![loss.4](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q4\loss.4.png)

   - Accuracy VS. Epochs

     ![accuracy.4](C:\Users\gao46\Documents\Linux\Workplace\CS690DL\ANS1\q4\accuracy.4.png)

The observations are same as Q3.