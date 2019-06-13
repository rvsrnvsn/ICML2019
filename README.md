# Notes from ICML 2019

### Summary
+ 2 sessions of 3-4 parallel tutorial tracks (Mon)
+ 3 sessions of 9 parallel conference tracks w/two 20-minute talks and eight 5-minute talks each (Tue-Thu)
  - ~300 posters (one per talk that day) in evenings
+ 15 parallel workshops each day (Fri-Sat)
+ 3300+ submissions, ~800 accepted papers
+ 5000+ attendees?
+ Videos of talks available at https://www.facebook.com/icml.imls/


### Tips and tricks
+ Whova app is very useful to plan attendance and keep up to date on conference events (as well as social events)
+ Choose tracks to attend based on 20-minute talks, try to sample interesting tracks rather than stay in one the entire day
+ Use 5-minute talks to determine initial set of posters to attend




## Monday, June 10

### Meta-Learning: from Few-Shot Learning to Rapid Reinforcement Learning **
_Chelsea Finn, Sergey Levine_

+ [N: Fill in notes.]

#### References
+ https://sites.google.com/view/icml19metalearning
+ http://tinyurl.com/icml-meta-slides


### A Tutorial on Attention in Deep Learning *
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
+ Simplifying computations using attention
  - Can use quaternions (hypercomplex numbers) to reduce parameters by 75% in some applications
  - Sparse transformer model
+ Open questions
  - Theory: function complexity, convergence analysis (a la Watson-Nadaraya estimator), regularization
  - Interpretation: Attention vs. meaning, how can we guide and what can we learn from multiple steps of attention/reasoning?
  - Computation: Dealing with large state spaces, memory footprint

#### References
+ http://alex.smola.org/talks/ICML19-attention.pdf
+ https://d2l.ai




## Tuesday, June 11

### The U.S. Census Bureau Tries to be a Good Data Steward in the 21st Century **
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
+ Science and policy must answer the following hard questions, among others
  - What should the privacy loss policy be for all uses of the 2020 Census?
  - How should the privacy loss budget be spent over the next 7 decades?
    * [N: Why 7 decades? Average lifespan of a person?]
+ [N: Look up papers by Abowd in AEI (or corresponding ArXiv articles) to understand better how $\epsilon$ in differential privacy can be interpreted for practical use in measuring privacy loss.]


### Best Paper: "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations" **
_Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem_

+ Disentangled representations
  - Aim: find disentangled representation without access to labels (ground-truth factors)
  - Theoretical result: For arbitrary data, unsupervised learning of disentangled representations is impossible!
    * Large-scale experimental study: Can we learn disentangled representations without looking at labels?
+ Takeaways
  - Role of inductive biases and supervision should be made explicitly
  - Concrete practical benefits of disentanglement should be demonstrated
  - Sound, reproducible experimental setup with several datasets is crucial
+ Poster 271
+ https://github.com/google-research/disentanglement_lib/
  

### Manifold Mixup: Better Representations by Interpolating Hidden States *
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


### Multi-Agent Adversarial Inverse Reinforcement Learning *
_Lantao Yu, Jiaming Song, Stefano Ermon_

+ Performance of RL agents heavily relies on quality of reward functions
  - Hard to fine-tune by hand
  - Solution: learn from expert demonstrations!
  - Imitation learning does not recover reward functions, so need to use IRL
+ Why reward learning?
  - Inferring intent, etc.
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
  - Agents should be able to cope with both discrete and continuous changes to data distribution and have no prior knowledge of when changes occur
+ Learn a sequence of policies that are somehow linked through distillation?


### Off-Policy Deep Reinforcement Learning without Exploration
_Scott Fujimoto, David Meger, Doina Precup_

+ Two agents trained with same off-policy algorithm on the same dataset can have vastly different performances based on whether they interact with the environment (i.e., whether they have access to state-action pairs they haven't visited)
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
  - Different learning rate for generator and discriminator
+ Convolutions excel at synthesizing image classes with few structural constrains, but fail to capture geometric or structural patterns
  - For example, images of dogs should generally have four legs
+ Self-attention can capture long-range dependencies more efficiently
  - Here, use softmax for attention weights over locations
  - This allows generator to allocate attention according to similarity of color and texture
  - Adjacent query points may result in very different attention maps
+ Rather than training discriminator multiple steps for each step training the generator, everything works better if you use a lower learning rate for the generator instead
+ Poster 11
+ https://github.com/brain-research/self-attention-gan


### High-Fidelity Image Generation With Fewer Labels *
_Mario Lucic, Michael Tschannen, Marvin Ritter, Xiaohua Zhai, Olivier Bachem, Sylvain Gelly_

+ BigGAN (Brock 2019) can produce photo-realistic images but is class-conditional
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
  - Originally came from imitation learning literature
  - Learner makes a decision, tries a policy and measure loss according to loss function, then determine whether to update policy
  - Policy loss is _predictable_ online learning [N: See slides for definition of predictability.]


### Learning a Prior over Intent via Meta-Inverse Reinforcement Learning **
_Kelvin Xu, Ellis Ratner, Anca Dragan, Sergey Levine, Chelsea Finn_

+ Motivation: A well-specified reward function remains an important assumption for applying RL in practice
  - Often easier to provide expert demonstrations and learn using IRL
  - Requires a lot of data to learn a generalizable reward
  - How can agents infer rewards from one or a few demonstrations?
  - Intuition: demos from previous tasks induce a prior over current ones
+ New algorithm MandRIL builds on model-agnostic meta-learning (MAML)
  - Embeds deep MaxEnt IRL into MAML
+ Poster 222


### DeepMDP: Learning Continuous Latent Space Models for Representation Learning
_Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, Marc Bellemare_

+ Motivation: develop simple representations for RL
+ DeepMDP: Learn NN representation for MDP and enforce reward and transition losses to keep simple model close to real one
+ Poster 108


### Importance Sampling Policy Evaluation with an Estimated Behavior Policy *
_Josiah Hanna, Scott Niekum, Peter Stone_

+ How can RL agents get the most from small amounts of experience?
  - Importance sampling for the RL sub-problem of policy evaluation
+ Regression importance sampling
  - Correction from empirical distribution (rather than behavior policy) to target policy
+ First to show that using an estimated behavior policy improves importance sampling in multi-step environments


### Learning from a Learner
_Alexis Jacq, Matthieu Geist, Ana Paiva, Olivier Pietquin_

+ Observe sequence of improving trajectories, try to learn optimal behavior by watching others learning
+ Poster 110


### Graph Matching Networks for Learning the Similarity of Graph Structured Objects *
_Yujia Li, Chenjie Gu, Thomas Dullien, Oriol Vinyals, Pushmeet Kohli_

+ Graph structured data appear in many applications, including programs (code search) and binaries (identifying software vulnerabilities)
+ How to find similar graphs from a large database?
  - Graph structures can vary a lot, nodes and edges can have attributes, need to reason about structure and semantics, and notion of similarity varies across problems
+ Binary function similarity search problem conducted with security researcher at Google
  - Binary analysis of executable
  - Look at control flow graph (graph sizes from 10-10^3), search in library of binaries with known vulnerabilities
  - Need to jointly reason about graph structure and graph semantics (node and edge attributes)
+ Two main approaches: graph embedding (hashing) and graph matching
  - First measures distance based on descriptors, while second measures distance based on graph itself
  - Can learn using supervised learning on labeled pairs or triplets
  - Want embedding function to preserve distance between graphs themselves
+ Matching model does better than embedding model for several applications such as binary function similarity search
+ Takeaways: learned approaches better than hand-engineered solutions, matching better than embedding alone, joint modeling of structure and features better than structure alone, performance better with more propagation steps
+ Learned attention patterns are often interpretable


### Shallow-Deep Networks: Understanding and Mitigating Network Overthinking *
_Yigitcan Kaya, Sanghyun Hong, Tudor Dumitras_

+ Do deep neural networks overthink?
  - Build internal classifiers to observer whether earlier layers can predict labels
  - How to train accurate internal classifiers without hurting accuracy of original network?
  - Use classification confidence of internal classifiers to determine where to cut off computation
  - Can quantify destructive effects of overthinking by looking at confusion matrix of internal classifier output vs. original architecture output
  - Can lead to significant boost in accuracy
+ Poster 24


### Discovering Options for Exploration by Minimizing Cover Time *
_Yuu Jinnai, Jee Won Park, David Abel, George Konidaris_

+ Introduce an objective function for exploration: cover time
  - Can try to minimize cover time by adding options to the agent
  - Minimization is NP-hard, so try to minimize upper bound on cover time instead
+ Algorithm
  - Embed state-space graph to real value (Fielder vector)
  - Generate options to connect two most distance states in this Fielder vector space
+ Poster 117


### Action Robust Reinforcement Learning and Applications in Continuous Control
_Chen Tessler, Yonathan Efroni, Shie Mannor_

+ Robust MDPs needed in practice (abrupt disturbances, model uncertainty)
  - Action robust MDPs, which consider uncertainty in actions
+ Takeaways
  - Robustness enables coping with uncertainty and transfer to unseen domains
  - Use gradient based approached for robust RL with convergence guarantees
  - Does not require explicit definition of uncertainty set
+ Poster 272


### The Value Function Polytope in Reinforcement Learning **
_Robert Dadashi, Marc Bellemare, Adrien Ali Taiga, Nicolas Le Roux, Dale Schuurmans_

+ Geometry of the space of possible value functions for a given MDP?
  - The ensemble of value functions is a possibly non-convex polytope
  - Can visualize convergence of different methods in value function space
+ Poster 119




## Wednesday, June 12

### Machine Learning for Robots to Think Fast **
_Aude Billard_

+ Consider settings where robot has to take action quickly, e.g., catching a ball from the air
+ Need a closed-form solution in order to be able to do computation fast enough (rather than optimizing some objective)
  - Consider the dynamical system $\dot{x} = f(x)$: how to learn $f$?
  - To learn a stable dynamical system, need to learn Lyapunov function $V$
  - Generate estimate $\dot{x} = f(x) = E[p(\dot{x}|x)]$ using probabilistic model $p(\dot{x}|x)$, giving a closed-form solution!
    * The training data in this case come from human demonstrations
    * If no demonstrations, can be combined with reinforcement learning
+ What if we're not having the robot catch a simple round ball, but a more complex object that requires us to choose the size of the grip?
  - This corresponds to a multi-attractor dynamical system
  - Use support vector regression to obtain estimate for $f(x)$
+ This works if object is not executing more complex dynamics, like spinning in the air
  - But can still adapt on-line to object dynamics by learning physics of object from demonstrations, then estimating most likely region to catch object
  - Showed video demonstrating approach by robot arm catching rotating tennis racquet thrown at it, and bottles half-filled with water
    * Traditional control theory approach would not be able to handle the half-filled bottle scenario, showing the power of machine learning approach
+ Bimanual problem
  - Learn bimanual workspace offline (using simulation environment for robot arms) in order to avoid collisions of arms at runtime
    * Feasible to safe learning on real robot arms to accomplish same task
  - Showed video of collision avoidance with human arms while navigating through bimanual workspace (i.e., task of avoiding human arms while preventing self-collision)
+ Modeling manipulations on deformable objects
  - Tasks include scooping out a melon, peeling a zucchini, etc.


### Test of Time Award: "Online Dictionary Learning for Sparse Coding"
_Julian Mairal, Francis Bach, Jean Ponce, Guillermo Sapiro_

+ Problem of matrix factorization $X = DA$ where the dictionary matrix $D$ is small and one may want sparsity of the factor matrix $A$, the dictionary matrix, or both (or some other structure that can be enforced by a regularization term in the loss function)
  - This covers non-negative matrix-factorization, sparse coding, non-negative sparse coding, structured sparse coding, approximation to sparse PCA, etc.
+ Alternating minimization of loss function on $D$ and $A$ worked quite well, but started experimenting with SGD and developed online dictionary learning approach
  - Impact due to the fact that datasets were becoming larger, and there was a sudden need for more scalable matrix factorization methods
  - An efficient software package (SPAMS toolbox)
  - Method was flexible in constraint/penalty design
+ Other thoughts
  - Simplicity is key to interpretability and model/hypothesis selection
  - Next form will probably be different from $l_1$, but which one?
  - Simplicity is not enough, various forms of robustness and stability are also needed


### Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning ***
_Natasha Jaques, Angeliki Lazaridou, Edward Hughes, Caglar Gulcehre, Pedro Ortega, DJ Strouse, Joel Z Leibo, Nando de Freitas_

+ Honorable Mention for ICML 2019 Best Paper Award
+ Intrinsic motivation for deep RL is still an unsolved problem
  - Curiosity, empowerment, etc.
+ Social learning is incredibly important
  - Social learning theory and ties to cultural evolution
  - How to let agents learn socially from other agents in multi-agent setting?
+ Give agents intrinsic social motivation for having causal influence on other agents' actions
  - Don't need to observe other agents' reward
  - Intuition is that this is equivalent to rewarding mutual information between agents' actions
+ Influence is via prediction of others' actions, which allows for computation of counterfactual actions
  - Sample counterfactual actions, then marginalize out causal influence of agent A on agent B
  - In expectation, this influence is the mutual information between agents' actions (novel form of social empowerment)
+ One issue is that influence doesn't necessarily have to be _positive_
  - Tested this in sequential social dilemmas (SSDs)
  - Cooperation is hard, but adding in influence reward helps achieve better collective rewards--but why?
    * Influence leads to emergent communication (for example in river cleanup environment)
+ Training communication via influence
  - Cheap talk between self-interested agents doesn't work, but adding in influence reward helps
  - Being influenced gets you a higher individual reward (i.e., useful information being communicated)
  - Communication appears to be about what the agent is doing in the environment, but still more work to be done to see what protocol these agents are learning
+ Conclusions
  - Whether action-action influence is prosocial depends on the environment
    * For self-interested agents communication may be the cheapest way to influence
  - Influence on self-interested agents via cheap talk communication must benefit the listener (for agents interacting repeatedly)
  - Social influence is a unified methods for promoting coordination and communication in multi-agent settings
  - Learn socially, train independently: does not require access to other agent's rewards or a centralized controller 


### Imitating Latent Policies from Observation *
_Ashley Edwards, Himanshu Sahni, Yannick Schroecker, Charles Isbell_

+ Imitation learning enables learning from state sequences (e.g., learning from videos)
  - Typical approaches need extensive interactions with environment, but humans can learn policies just by watching
+ Approach: Learn latent policy from observations
  - Given sequence of noisy expert observations, learn _latent_ policy
  - Use a few environment steps to align actions
  - This is done using a generative model
+ This can outperform Behavioral Cloning from Observation (BCO)
+ Poster 33


### SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning
_Marvin Zhang, Sharad Vikram, Laura Smith, Pieter Abbeel, Matthew Johnson, Sergey Levine_

+ How to do efficient RL from images?
  - LQR-FLM fits local models for policy improvement, not forward prediction
    * Fits linear dynamics and quadratic cost for policy improvment
  - Enable LQR-FLM for images using structured representation learning, leading to SOLAR
+ SOLAR more efficient than both prior model-free and model-base methods
  - Can transfer a representation and model to multiple arm positions in block stacking
+ Poster 34


### Estimating Information Flow in Deep Neural Networks **
_Ziv Goldfeld, Ewout van den Berg, Kristjan Greenewald, Igor Melnyk, Nam Nguyen, Brian Kingsbury, Yury Polyanskiy_

+ Macroscopic understanding of deep learning
  - Consider deterministic feedforward DNN
+ Information bottleneck theory (Shwartz & Tishby '17)
  - Training comprises of two phases: (1) short phase of fitting (where mutual information between intermediate layers and inputs rise) and (2) long phase of compression (where mutual information slowly drops)
  - But one can show that for DNNs with strictly monotone nonlinearities, MI is independent of the DNN parameters!
    * Vacuous mutual information and mis-estimation
+ Instead, move to a slightly different framework of a noisy DNN (inject small Gaussian noise to neurons' output), so no longer deterministic
  - Unfortunately, analysis shows that sample complexity of any accurate estimator is exponential in the dimension of the input vector, so can only work with small-scale experiments
  - Can show that compression (and subsequent drop in mutual information) driven by clustering of representations


### Characterizing Well-Behaved vs. Pathological Deep Neural Networks
_Antoine Labatie_

+ There is no mature theory able to validate the fill choice of hyperparameters leading to SOTA performance in DNNs
  - Much research aimed at this theory has focused on networks at the time of random initialization
+ Considered the following methodologies
  - Simultaneous propagation of signal and additive noise corrupting the signal
  - Data randomness (measured using effective rank and normalized sensitivity)
  - Model parameter randomness
+ Applied these methodologies in order to determine that feedforward nets are pathological at high depth, while batch-normalized ResNets are well-behaved


### Traditional and Heavy-Tailed Self Regularization in Neural Network Models *
_Michael Mahoney, Charles H Martin_

+ Why does deep learning work, and how to use insights to improve engineering of DNNs?
+ Consider energy landscape of neural network and gradually add regularization to study weight matrix
  - Look at empirical spectral density
  - Studied using heavy-tailed random matrix theory
    * Spectrum looks heavy-tailed in almost all applications
  - `pip install weightwatcher` to look at spectral density of your DNN


### The Natural Language of Actions **
_Guy Tennenholtz, Shie Mannor_

+ The meaning of a word comes from its context (semantics)
+ Actions are characterized by the company they keep
  - Context is generated by an optimal agent (sufficient but not necessary), which demonstrates acceptable behavior in the environment
+ Act2Vec
  - Given demonstrations from optimal agents, develop contexts from demonstration trajectories and feed into skip gram as in Word2Vec
  - Leads to partitioning of action space that helps understanding of how to move in the environment
+ Can be used in RL framework to represent the Q-function (proximity in embedding space corresponds to similar Q-values)
  - Also useful for exploration, by dividing action embedding space into clusters, sample a cluster uniformly, then sample an action from that cluster
+ Applied to StarCraft II
  - Looked at 2B actions (100 years (?) of human gameplay)
  - Action space nearly partitions into different clusters, some of which correspond to individual strategies
+ Poster 41


### Control Regularization for Reduced Variance Reinforcement Learning
_Richard Cheng, Abhinav Verma, Gabor Orosz, Swarat Chaudhuri, Yisong Yue, Joel Burdick_

+ RL methods suffer from high variance in learning
  - On the plus side, can allow is to optimize policy with no prior information (only sampled trajectories from interactions)
  - Is this necessary or desirable?
+ Regularization with a control prior
  - Stronger regularization constrains exploration
  - Takeaway: Control regularization can reduce variance, lead to higher rewards and faster learning, and help with safety
+ Poster 42


### Trajectory-Based Off-Policy Deep Reinforcement Learning
_Andreas Doerr, Michael Volpp, Marc Toussaint, Sebastian Trimpe, Christian Daniel_

+ How far can we push "model-free" RL?
  - Problems of data inefficiency, gradient variance, exploration vs. exploitation tradeoff
  - Developed deep deterministic off-policy gradients (DD-OPG) to incorporate data-efficient sample reuse, low-noise deterministic rollouts, with lengthscale in action space as the only model assumption
+ Poster 44


### A Deep Reinforcement Learning Perspective on Internet Congestion Control **
_Nathan Jay, Noga H. Rotman, Brighten Godfrey, Michael Schapira, Aviv Tamar_

+ Internet congestion control determines quality of experience on the internet
  - QUIC by Google
  - New model similar to PCC (use network statistics to determine how to send information)
+ [N: Looks like a very interesting application of deep RL. Can one derive a multiagent learner version (that induces cooperation)?]
+ https://github.com/PCCProject/PCC-RL
+ Poster 45


### Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations **
_Daniel Brown, Wonjoon Goo, Prabhat Nagarajan, Scott Niekum_

+ Leveraging ranked demonstrations to learn reward using binary classifier
  - Scalable, unlike some other approaches
  - Can learn to accomplish tasks better than best human demonstration


### Distributional Multivariate Policy Evaluation and Exploration with the Bellman GAN
_Dror Freirich, Tzahi Shimkin, Ron Meir, Aviv Tamar_

+ Distributional Bellman operator can be seen as a form of a GAN
  - Can use GAN-based algorithm for distributional RL


### Do ImageNet Classifiers Generalize to ImageNet? **
_Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, Vaishaal Shankar_

+ How reliable is current ML?
  - What should test error imply? At the least, generalization
+ Built a new test set for ImageNet in which the same classifier was tested on multiple test sets, saw substantial drop in performance
  - Could be due to overfitting through test set reuse, or by distribution shift (which caused the drop in performance)
  - Original test accuracy predicts new test accuracy (relationship appears to be linear)
+ Replicated ImageNet dataset creation process to create new test sets
  - No signs of adaptive overfitting on CIFAR-10 and ImageNet (despite 10 years of reuse)
    * Relative ordering preserved--progress is real!
    * Similar conclusions for MNIST and Kalle
  - Distribution shifts are a real problem, even in a carefully-controlled reproducibility experiment
+ Questions
  - Does robustness to adversarial inputs (e.g., image perturbations) also protect against distribution shift?
  - Can we characterize our distribution shifts?
  - What is human accuracy on our test sets?
  - Why is there no adaptive overfitting
  - Need more diverse and challenging test sets
+ Poster 141


### Exploring the Landscape of Spatial Robustness *
_Logan Engstrom, Brandon Tran, Dimitris Tsipras, Ludwig Schmidt, Aleksander Madry_

+ Adversarial perturbations have generally been considered small in $l_1$ norm
  - How about perturbations like small rotations, which are not small in any $l_p$ norms?
  - Spoiler: models are not robust
+ Can use robust optimization to train on perturbed inputs in order to develop spatial defenses?
  - How to find worst-case translations, rotations?
  - First-order methods don't work, but exhaustive search (which is feasible) does
  - Leads to much better accuracy for CIFAR-10 and ImageNet in adversarial setting
  - Intuitions from $l_p$ robustness do not necessarily transfer
+ Poster 142


### Sever: A Robust Meta-Algorithm for Stochastic Optimization
_Ilias Diakonikolas, Gautam Kamath, Daniel Kane, Jerry Li, Jacob Steinhardt, Alistair Stewart_

+ Can you learn a good classifier from poisoned training data?
  - Assume $\epsilon$-fraction are adversarially corrupted
+ Framework for robust stochastic optimization called SEVER
  - Until termination:
    * Train black-box learner to find approximate minima of empirical risk on corrupted training set
    * Then run outlier detection on gradients of loss function (to detect those inputs that were likely adversarial?)
+ Poster 143


### Analyzing Federated Learning through an Adversarial Lens
_Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, Seraphin Calo_

+ Federated learning is distributed learning in which agent perform local training (on locally available data) and send model updates to global server which aggregates this information and distributes it back to agents
+ Threat model
  - Single malicious agent
  - No access to current updates from other agents
  - Attacks with respect to previous global state
  - Aim: Cause misclassification
+ Takeaways
  - Stealthy model poisoning can be done
  - Detection strategies make attacks more challenging but can be overcome by white-box attackers


### Fairwashing: The Risk of Rationalization *
_Ulrich Aivodji, Hiromi Arai, Olivier Fortineau, Sébastien Gambs, Satoshi Hara, Alain Tapp_

+ LaundryML: Systematically generating fake explanations
  - Can develop explanations for decisions that appear to be fair but actually aren't
  

### Understanding the Origins of Bias in Word Embeddings *
_Marc-Etienne Brunet, Colleen Alkalay-Houlihan, Ashton Anderson, Richard Zemel_

+ Word embeddings can biased in the concepts they implicitly learn
  - How to measure bias in word embeddings?
    * Implicit Association Test (IAT)
+ Compute differential bias of documents used to train the model
  - Applied influence functions to GloVe
+ Bias can be quantified and correlates with known human biases
  - Can identify the documents that most affect bias, and estimate their impact
+ Future work
  - Consider multiple biases simultaneously
  - Use metrics that depend on more words
  - Consider bias in downstream tasks where embeddings are used
  - Does this approach carry over to BERT and other language models?




## Thursday, June 13

### Towards a Deep and Unified Understanding of Deep Neural Models in NLP
_Chaoyu Guan, Xiting Wang, Quanshi Zhang, Runjin Chen, Di He, Xing Xie_
+ Considering coherency and generality in deep neural model in NLP using information theoretic approach
+ Conclusoin: BERT and Transformer use words for prediction while LSTM and CNN use subsequences of sentences for prediction
+ Poster 62


### Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley Value Approximation
_Marco Ancona, Cengiz Oztireli, Markus Gross_

+ Evaluating attribution methods
+ Shapley values is the only attribution method that satisfies desirable set of properties
  - The average marginal contribution of a feature
  - Shapley value sampling is unbiased but may require many samples. Can we do better than sampling?
+ New method DASP sits between gradient-based methods (which are fast but give poor estimates) and sampling methods


### Functional Transparency for Structured Data: a Game-Theoretic Approach
_Guang-He Lee, Wengong Jin, David Alvarez-Melis, Tommi Jaakkola_

+ Methods for post-hoc explanation usually involve training simple approximation (local model) to DNN such as linear model or decision tree
+ How to train complex models to exhibit meaningful properties locally?
  - Use witness model to enforce desirable properties
+ Poster 64


### Exploring interpretable LSTM neural networks over multi-variable data
_Tian Guo, Tao Lin, Nino Antulov-Fantulin_

+ Multi-variable time series
  - Could concatenate input, but will end up giving one joint hidden state in RNN
+ How to get prediction model that is accurate and interpretable?
  - Keep separate hidden states, but use two attention mechanisms, one to combine hidden states of variables and one to track temporal states


### TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing **
_Augustus Odena, Catherine Olsson, David Andersen, Ian Goodfellow_

+ How to test and debug neural networks?
  - No crashes, no NaNs, same answer for mutated image if difference is small, same answer from quantized/non-quantized network, saem answer from base and refactored network
  - NNs are programs, so lets use coverage-guided fuzzing
+ 2 big fuzzers: AFL and libFuzzer
  - Maintain corpus of inputs, mutate elements in that corpus, and put new mutations in corpus if they exercise new coverage (coverage being that basic blocks are hit)
    * Used in security community and have found many security flaws (particularly AFL)
+ How to fuzz NNs?
  - Need to define mutation, guidance (coverage for a DNN), and errors
  - Use Gaussian noise for mutations, sometime with constraints to ensure image remains in domain, etc.
  - Code coverage does not apply well to DNNs
    * All examples trigger identical code paths, only the data differs
  - Contribution: use nearest neighbor algorithm on activations
    * If mutated input gives activations close enough to activations already seen, say that input is covered
    * TensorFuzz consider input "interesting" if distance is greater than some threshold
    * Technique is practical and computationally feasible
  - Use property base testing (PBT) for errors
  - Important detail is how to choose inputs
    * Bias to recently added elements (depth-first search), use nearby inputs until mutations no longer interesting
    * Otherwise use breadth-first search
+ TensorFuzz results
  - Finds NaNs faster than baseline random search
  - Surfaces quantization errors
  - Facilitates refactoring
    * Key technique during refactoring: fuzz difference between new code and old code
    * Example: inefficient flipping in Tensorflow random flip (sped up flipping by 2.6x to 45x with 6 line code change)
  - Has found bugs in open-source code


### State-Regularized Recurrent Neural Networks
_Cheng Wang, Mathias Niepert_

+ What exactly do RNNs learn on sequential inputs?
  - Tend to generalize poor on longer sequences
+ Approach: learn a finite set of states (k centroids) and force RNNs to operate like automata with external memory
+ https://github.com/deepsemantic/sr-rnns
+ Poster 68


### On the Connection Between Adversarial Robustness and Saliency Map Interpretability *
_Christian Etmann, Sebastian Lunz, Peter Maass, Carola-Bibiane Schönlieb_

+ Models trained to be more robust to adversarial attacks tend to yield more interpretable saliency maps, but why?
  - There is higher visual alignment between image and saliency map for models trained against attacks (perturbations to inputs)
+ Gives theoretical backing to this connection
+ Poster 70


### Analogies Explained: Towards Understanding Word Embeddings **
_Carl Allen, Timothy Hospedales_

+ Honorable Mention for ICML 2019 Best Paper Award
+ Problem: linking semantics to geometry for word embeddings
+ Word2Vec
  - $W^T C \simeq PMI - log k$ where $W$ is word weight matrix, $C$ is context weight matrix, $k$ is number of negative samples, and PMI is pointwise mutual information
+ Look at PMI vectors more closely (summing PMI vectors of a paraphrase)
  - PMI of paraphrase is sum of PMI of individual words plus three error terms that can be interpreted
  - For example, if one word set is W = {woman, king} and another is W_\* = {man, queen}, does W paraphrase W_\*? Can measure this now
  - Leads to word transformations, in which one word is transformed to another word if there are additional words that can be added to each so that the resulting word sets paraphrase each other
    * [N: Is paraphrasing a commutative operation?]
+ Poster 101


### Parameter-Efficient Transfer Learning for NLP
_Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly_

+ Transfer learning for NLP
  - Take a large pretrained model (BERT) and fine-tune
+ Main idea: instead of fine-tuning for each new task, train tiny adapter modules at each layer for each task
  - Gets similar performance with many fewer parameters (30x reduction at only 0.4% accuracy drop)
+ Poster 102


### Efficient On-Device Models using Neural Projections *
_Sujith Ravi_

+ Motivation: how to get success of big NNs running on cloud to tiny NNs running on-device?
  - Need on-device computation for privacy, limited connectivity, efficient computing, consistent experience
  - Already in use in some Google products
+ Can do this using neural projections which are very fast and small NNs
  - Transform weights to compact bit vectors (locality sensitive ProjectionNets)
+ Build your own on-device models


### Improving Neural Language Modeling via Adversarial Training
_Dilin Wang, Chengyue Gong, Qiang Liu_

+ Language models tend to overfit
+ Main idea: inject adversarial perturbation on the word embedding vectors in the softmax layer and maximize worst case performance
  - Has closed-form solution and is easy to implement in practice
  - Adversarial MLE promotes diversity
  - Improves language modeling empirically
+ Poster 105


### Mixture Models for Diverse Machine Translation: Tricks of the Trade *
_Tianxiao Shen, Myle Ott, Michael Auli, Marc'Aurelio Ranzato_

+ Translation is a one-to-many mapping (a sentence can have different translations)
  - How to efficiently decode a diverse set of hypotheses?
+ Search for multiple modes is difficult in neural machine translation
  - Beam search can effectively find one likely translation but cannot explore multiple modes
+ Models explored
  - Introduced a latent variable (conditional VAE) to explicitly model uncertainty over potential translations
    * Produced translations of high quality, but low diversity
  - Worked with hard and soft mixture models
    * Required careful design in order to produce multiple hypothesized translations of reasonable quality and diversity
  - Used a metric similar to BLEU to measure this
+ https://github.com/pytorch/fairseq
+ Poster 106


### MASS: Masked Sequence to Sequence Pre-training for Language Generation
_Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu_

+ BERT and GPT are suboptimal on sequence-to-sequence based language generation tasks
  - BERT can only be used to pretrin encoder and decoder sepearately
  - encoder to decoder attention is very important, which BERT does not pretrain
+ New method MASS proposed to jointly pre-train encoder and decoder via attention module
  - Forces encoder to extract meaning information from sentence
  - Develop the decoder with the ability of language modeling
+ Poster 107


### Humor in Word Embeddings: Cockamamie Gobbledegook for Nincompoops **
_Limor Gultchin, Genevieve Patterson, Nancy Baym, Nathaniel Swinger, Adam Kalai_

+ Single-word humor (individual words that are funny)
  - Original data collection from Mechanical Turk (e.g., is "yadda yadda": funny-sounding, juxtaposition, sexual, etc.)
+ Questions considered
  - Can we use word embeddings to capture humor theories and humor direction?
  - Can we identify different sense of humor across demographic groups?
  - Can we define individual senses of humor and predict users' tasts?
+ [N: Funny and interesting results, particularly regarding humor across demographic groups.]
+ Poster 108


### CHiVE: Varying Prosody in Speech Synthesis with a Linguistically Driven Dynamic Hierarchical Conditional Variational Network
_Tom Kenter, Vincent Wan, Chun-an Chan, Robert Clark, Jakub Vit_

+ Modeling intonation in prosody
  - Use conditional VAE to capture different intonations
+ Language has hierarchical linguistic structure (sentence, words, syllables, ...)
  - Use linguistic knowledge in building network architecture
+ Poster 110
