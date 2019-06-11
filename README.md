# Notes from ICML 2019

#### Summary and takeaways
+ 3-4 parallel tutorial tracks (Mon)
+ 9 parallel conference tracks w/two 20-min talks and ten 5-min talks each (Tue-Thu)
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

### The U.S. Census Bureau Tries to be a Good Data Steward in the 21st Century 
_John Abowd_

+ Census Bureau has dual mandate: (1) collect all data necessary to underpin our democracy, (2) protect privacy of individual data to ensure trust and prevent abuse
+ 2020 Census has a cost estimate of $15.6B dollars (largest single peacetime expenditure)
  - Major data products include apportion for the House of Representatives, supply data to all state redistricting offices, demographic and housing characteristics, detailed race and ethnicity data, and native peoples data
  - For 2010 Census, this was >150B statistical tabulations compiled from 15GB data
  - Estimate 140M addresses, 126 occupied housing units, and 330M people
  - Generous estimate is ~100GB of data from 2020 Census (which is equivalent to <1% of worldwide mobile data use/second)
+ Database reconstruction vulnerability
  - Lessons from cryptography: can't publish too many statistics, noise infusion is necessary, transparency about methods is a benefit
  - Tried database reconstruction on all persons in 2010 Census to see how vulnerable current process is
    * Census block, sex, age (+/- 1 year), race, ethnicity reconstructed exactly for 71% of population
    * Linked to commercial data to personally re-identify 17% of total population (i.e., leaked PII)
  - Moral: comparing common features allows highly reliable entity resolution, which can be made much harder with provable privacy guarantees!
  - Law of information recovery imposes a resource constraint on the publication of data
    * Fundamental tradeoff between accuracy and privacy loss (i.e., production possibility frontier in economics)
    * It is _infeasible_ to operate above the frontier---therefore, want to operate on the frontier curve (research can move this frontier out)
+ Differential privacy in practice
  - Statistical data is subject to a shared privacy loss budget
  - It can be shown that published data for redistricting and many other statistical products fit within this framework
+ Science and policy must answer the following hard questions, aong other
  - What should the privacy loss policy be for all uses of the 2020 Census?
  - How should the privacy loss budget be spent over the next 7 decades?
    * [N: Why 7 decades? Average lifespan of a person?]
+ [N: Look up papers by Abowd in AEI (or corresponding ArXiv articles) to understand better how $\epsilon$ in differential privacy can be interpreted for practical use in measuring privacy loss.]


### Best Paper: "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"
_Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem_

+ Disentangled representations
  - Aim: find disentangled representation without access to labels (ground-truth factors)
  - Theoretical result: For arbitrary data, unsupervised learning of disentangled representations is impossible!
    * Large-scale experimental study: Can we learn disentangled representations without looking at labels?
+ Takeaways
  - Role of inductive biases and supervision should be made explicity
  - Concrete practival benegits of disentanglement should be demonstrated
  - Sound, reproducible experimental setup with several datasets is crucial
+ Poster 271
+ https://github.com/google-research/disentanglement_lib/
  

### Manifold Mixup: Better Representations by Interpolating Hidden States
_Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, David Lopez-Paz, Yoshua Bengio_

+ Problem with current deep networks is they are overly confident in their estimates
+ Algorithm tries to alleviate this by mixing between two random examples from the minibatch at randomly selected layer
  - On each update, select random layer uniformly (including the input)
  - Sample Beta random variable $\lambda$, mix between two random examples from the minibatch at the selected layer with weights $\lambda# and $(1-\lambda)$
  - Mix labels for those two examples accordingly
+ Can help improve likelihood and helps with cases with little labeled data


### Online Meta-Learning
_Chelsea Finn, Aravind Rajeswaran, Sham Kakade, Sergey Levine_

+ Learn new task with few datapoints
+ How to do meta-learning with tasks given in sequence?
  - Start with slow learning, then transition to rapid learning
+ Learn sequence of tasks from non-stationary distribution


### Multi-Agent Adversarial Inverse Reinforcement Learning
_Lantao Yu, Jiaming Song, Stefano Ermon_

+ Performance of RL agents heavily relies on quality of reward functions
  - Hard to fine-tune by hand
  - Solution: learn from expert demonstrations!
  - Imitation learning does not recover reward functions, so need to use IRL
+ Why reward learning?
  - Infering intent, etc.
  - Reward function is the most succinct, robust, transferable representation of task
+ Single-agent IRL is ill-defined
  - MaxEnt IRL provides probabilistic framework to solve ambiguity
  - Adversarial IRL (AIRL) provides efficient sampling-based approximation to MaxEnt IRL
+ Multi-agent setting
  - Markov games instead of MDPs
  - Solution concepts now involve Nash equilibria (NE), but NE and correlated equilibria (CE) are incompatible with MaxEnt IRL!
  - Instead, propose logistic stochastic best response equilibrium (LSBRE), which is compatible


### Policy Consolidation for Continual Reinforcement Learning
_Christos Kaplanis, Murray Shanahan, Claudia Clopath_
+ How to deal with catastrophic forgetting? For RL, as tasks change the data distribution changes over time
  - Agents should be able to cope with both discrete and continuiius changes to data distribution and have no prior knowledge of when changes occur
+ Learn a sequence of policies that are somehow linked through distillation?


### Off-Policy Deep Reinforcement Learning without Exploration
_Scott Fujimoto, David Meger, Doina Precup_

+ Two agents trained with same off-policy algorthm on the same dataset can have vastly diferent performances based on whether they interact with the environment (i.e., whether they have access to state-action pairs they haven't visited)
  - Due to what can be called extrapolation error
+ New algorithm BCQ outperforms DDPG
+ Poster 38


### Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation
_Ruohan Wang, Carlo Ciliberto, Pierluigi Amadori, Yiannis Demiris_

+ Imitation learning: policy learning from a limited set of expert demonstrations
+ Replaces adversarial training for learning reward function with expert policy distillation step


### Self-Attention Generative Adversarial Networks
_Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena_

+ In order of importance, new implementation of GAN
  - Add self-attention blocks to generator and discriminator
  - Spectral normalization
  - Different learing rate for generator and discriminator
+ Convolutions excel at synthesizing image classes with few structural constrains, but fail to capture geometric or structural patterns
  - For example, images of dogs should generally have four legs
+ Self-attention can capture long-range dependencies more efficiently
  - Here, use softmax for attention weights over locations
  - This allows generator to allocate attention acording to similarity of color and texture
  - Adjacent query points may result in very different attention maps
+ Rather than training discriminator multiple steps for each step training the generator, everything works better if you use a lower learning rate for the generator instead
+ Poster 11
+ https://github.com/brain-research/self-attention-gan


### High-Fidelity Image Generation With Fewer Labels
_Mario Lucic, Michael Tschannen, Marvin Ritter, Xiaohua Zhai, Olivier Bachem, Sylvain Gelly_

+ BigGAN (Brock 2019) can produce photo-realistic images but is class-contitional
  - Can we do this without labels? How to close gap between conditional and unsupervised GANs?
+ Infer labels from self-supervised and semi-supervised approaches
+ Poster 13
+ https://github.com/google/compare_gan


### Revisiting precision recall definition for generative modeling
_Loic Simon, Ryan Webster, Julien Rabin_

+ Currently, FID (scalar value) is used to determine realism of generated image
+ How to better evaluate precision and recall? [N: See slides for precise definition.]


### Predictor-Corrector Policy Optimization
_Ching-An Cheng, Xinyan Yan, Nathan Ratliff, Byron Boots_

+ Cost of interactions > cost of computation
  - This motivates use of planning and better sample efficiency
  - Model-free (unbiased, inefficient) vs. model-based (efficient, biased) approaches
  - New algorithm PicCoLO tries to use best of both worlds
    * Main idea: don't fully trust a model, but only use the correct parts
+ Policy optimization as online learning
  - Learner makes a decision, tries a policy and measure loss according to loss function, then determine whether to update policy
  
