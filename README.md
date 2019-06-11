# Notes from ICML 2019

#### Summary and takeaways
+ 3-4 parallel tutorials (Mon)
+ 9 parallel conference tracks (Tue-Thu)
+ 15 parallel workshops (Fri-Sat)



## Monday, June 10

### Meta-Learning: from Few-Shot Learning to Rapid Reinforcement Learning
_Chelsea Finn, Sergey Levine_

+ [N: Fill in notes.]

#### References
+ https://sites.google.com/view/icml19metalearning
+ http://tinyurl.com/icml-meta-slides



### A Tutorial on Attention in Deep Learning
_Alex Smola, Aston Zhang_

+ Animal attention is resource saving and allows for variable state manipulation
+ Watson-Nadaraya estimator (weighted regression) $y = \sum_{i=1^M} \alpha(x,x_i) y_i$ with normalized Gaussian weights $\alpha(x,x_i)$ can be seen as a simple example of attention
  - Consistent and simple estimator
  - Deep learning analogue ("Deep Sets")
    * Learns the weighting function
    * Replaces averaging (i.e., pooling) $f(X) = \rho(\sum_{x \in X} \phi(x))$ with weighted pooling (i.e., attention pooling) $f(X) = \rho(\sum_{x \in X} \alpha(x,w) \phi(x))$
+ Attention is very useful to utilize context in NLP tasks
  - Attention/pooling doesn't help much in applications such as question answering since it misses intermediate steps
  - Iterative pooling helps with multi-step reasoning tasks by repeatedly updating the query based on the prior one
    * $q_{t+1} = \rho(\sum_{x \in X} \alpha(x,q_t) \phi(x))$
    * Practically, only iterate 2-3 times
  - Seq2Seq models fail for long sentences
    * Weighted iterative pooling that emit one symbol at a time can do much better
+ Neural Turing machine (NTM) incorporates memory with attention
+ Multi-head attention (self-attention when query, key, and value match)
  - Multi-head attention can be used for semantic segmentation, wherein pixels in an image can be classified based on their surrounding context
+ Transformer models
  - Bidirectional Encoder Representations from Transformers (BERT) can be used for two tasks: masked language models and next-sentence prediction
  - To contrast, Generative Pre-Training (GPT) model uses only the transformer _decoder_ and is uni-directional (only looks forward)
+ Simplying computations using attention
  - Can use quaternions (hypercomplex numbers) to reduce parameters by 75% in some applications
  - Sparse transformer model
+ Open questions
  - Theory: function complexity, convergence analysis (a la Watson-Ndaraya estimator), regularization
  - Interpretation: Attention vs. meaning, how can we guide and what can we learn from multiple steps of attention/reasoning?
  - Computation: Dealing with large state spaces, memory footprint
  
#### References
+ http://alex.smola.org/talks/ICML19-attention.pdf
+ https://d2l.ai



## Tuesday, June 11
