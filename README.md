# Minimizing search time for finding an effective treatment
Code for generating data, algorithms for creating policies and visualizing the results for the paper *Learning to search efficiently for causally near-optimal treatments* [link](https://arxiv.org/abs/2007.00973)  by Samuel Håkansson, Viktor Lindblom, Omer Gottesman and Fredrik D Johansson accepted for publication at NeurIPS 2020. Read **Evaluating policies and Plotting** for instructions of how to
run the project in the simplest way.

# Data Generation
Data generation is done by so called distributions found in DataGenerator/distributions.py. Each distribution implements
four methods that describe a patient and which treatments that patient receives. The default distribution used is 
DiscreteDistributionWithSmoothOutcomes for most other files. The DiscreteDistributon family of distributions has 
methods for printing statistics of the distributions as well.

To generate data, use the helper class DataGenerator/data_generator.py. This class has functions to generate both 
training and test data. Most other classes expect the training data to be in "split" format, this is achieved by calling
the split_patients function.

# Algorithms
The algorithms can be found in the Algorithms/ folder, including classes in Algorithms/Approximators/ for approximating
the probability of outcomes and classes in Algorithms/Constraints/ for calculating the constraints used.

**Algorithms**
*constrained_greedy.py* is the main greedy constrained variant, using a greedy rule to find the best treatment in each
trial.

*constrained_dynamic_programming.py* is the main dynamic programming variant.

*naive_greedy.py* is the greedy variant that does not use a constraint.

*naive_dynamic_programming.py* is the dynamic programming variant that does not use a constraint.

*emulated_doctor.py* emulates what the doctor would have done for the Antibiotic resistance data set.

*distribution_algorithm_wrapper.py* can be used to compare the policy that the distribution uses to the algorithmic 
ones.

**Approximators**

*statistical_approximator.py* uses a smoothing function that can either be 'gaussian' or 'none'. It estimates outcome 
probabilities using frequency statistics.

*function_approximator.py* uses a Random Forest regression to approximate the outcome probabilities.

*exact_approximator.py* uses the real probabilities from the distribution to calculate the outcome probabilities.

*doctor_approximator.py* is used by emulated_doctor.py.

**Constraints**

*better_treatment_constraint.py* can either use an 'upper' or 'lower' bound for quickly estimating the constraint.

*true_constraint.py* calculates the constraint exactly, but requires more processing time.

# Evaluating policies and plotting

Use the file *Main/Evaluate.py* to run custom evaluations.

To do an easier evaluation, the main way to do this is in Main/SingleEvaluations/ where we have four files that evaluate
the policies in different ways. To run an evaluation, first copy a Settings file in Main/SingleEvaluations/Settings.
Take a look at *GeneralDeltaSweepSettings.py* and *DataAmountSettings.py* for examples, but each file requires you to
edit the 'setup_algorithms' function to specify which algorithms to evaluate and the 'get_settings' function to set 
which other settings, e.g. number of data sets to average over, to use.

**Evaluators**

To use, simply set the setting you want to use in the 'load_settings' function and run.

*Antibiotics.py* does a single evaluation of the antibiotics data set.

*AntibioticsSweepDelta.py* evaluates the antibiotics data set over several different values of delta.

*SweepDataSizes.py* evaluates policies over different data set sizes.

*SweepDelta.pu* evaluates policies over different values of delta.

**Plotters**

To re-plot already evaluated policies, the plotters can be used. Can also plot Time vs Effect instead of separate plots.
Simply change the 'get_settings' function to load the correct settings file. The Antibiotics files are used to plot 
antibiotics data. The Split files are used to plot two separate images instead of one combined.

*PlotSweepData.py* plots a data set size sweep.

*PlotSweepDelta.py* plots a delta values sweep.

*PlotTimeVsEffect.py* plots the time against the effect for both data set sizes and delta. Works better for delta.
