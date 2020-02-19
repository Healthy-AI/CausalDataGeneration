# CausalDataGeneration
Generate causal data from a history of treatments

This package consists of two parts: 
a data generator and a q-learning model.

The data generator creates a population of patients from a tailored distribution (in distributions.py).
The variables for SimpleDistribution are: covariates X [0,1], confounders Z [0, 1, 2], treatments A [1,2,3,4], and outcomes Y [0,1,2].
Patients have a history which is a list of treatments and outcomes e.g. [(a0, y0), (a1, y1)]

The data is generated such that, for every patient, we draw confounder Z and then covariate X,
then we draw treatment A and outcome Y until all treatments are tried on the patient.
After that, we trim the data so that the patients stopped trying a new treatment if it found
 an outcome equal to 2.

There is also functionality to split the history of a patient. A patient with covariate x0 
and history [(a0, y0), (a1, y1)] will become three patients with covariate x0 and history [(a0, y0)], [(a1, y1)], and [(a0, y0), (a1, y1)]

 A simple offline q-learning algorithm is also implemented in q_learning.py. 
 Each final state, e.g. where a stopping action is taken, is initialized to the value of the best treatment at that time, 
 or -infinity if no medium or better treatment is found. 
 We use a discount factor of 1 since the problem has a finitie horizon, instead opting to have negative reward of -0.5 
 for each tried medicine. The value of the reward is not very important for this simple problem, and any negative reward 
 solves the problem.
Actions are sampled randomly from the provided dataset with replacement.
 
 