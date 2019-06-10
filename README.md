# ICML2019
Notes from ICML 2019

#### Summary
+ 3-4 parallel tutorials (Mon)
+ 9 parallel conference tracks (Tue-Thu)
+ 15 parallel workshops (Fri-Sat)


## Monday, June 10

### Meta-Learning: from Few-Shot Learning to Rapid Reinforcement Learning
_Chelsea Finn, Sergey Levine_

#### References
+ https://sites.google.com/view/icml19metalearning
+ http://tinyurl.com/icml-meta-slides


### A Tutorial on Attention in Deep Learning
_Alex Smola, Aston Zhang_

+ Animal attention is resource saving and allows for variable state manipulation
+ Watson Nadaraya estimator (weighted regression) $y = \sum_{i=1^M} \alpha(x,x_i) y_i$ with normalized Gaussian weights $\alpha(x,x_i)$ can be seen as a simple example of attention
  - Is consistent and simple
  - Deep learning analogue ("Deep Sets") learns the weighting function, and replaces averaging (i.e., pooling), with weighted pooling
+ Attention is very useful to utilize context in NLP tasks

#### References
+ http://alex.smola.org/talks/ICML19-attention.pdf
+ https://d2l.ai
