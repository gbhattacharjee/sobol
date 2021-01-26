Example problem
===============

Say we have a physical system whose performance, *y*, depends on three independent variables, x\
:sub:`1`, x\ :sub:`2`, x\ :sub:`3` that we model as taking values uniformly at random in the interval [*-\pi*, *\pi*].
We model the performance of the system as y = sin x\ :sub:`1` + a sin \ :sup:`2` x\ :sub:`2` + b x\
:sub:`3` :sup:`4` sin x\ :sub:`1`. (This is the Ishigami function.)

We would like to know which variable influences the
performance of the physical system the most. Why? Perhaps there's a certain range of *y* that is acceptable, and we
would like to set the values of x\ :sub:`1`, x\ :sub:`2`, x\ :sub:`3` such that *y* is always in that acceptable range.

One way to understand the relative influence of each input with respect to the output is to perform a sensitivity
analysis.

What is sensitivity analysis (SA)?
----------------------------------
Saltelli et al. (2004) define sensitivity analysis as "the study of how the uncertainty in the output of a model \dots
can be apportioned to different sources of uncertainty in the model input."

Local (or deterministic) SA is performed around a particular point of interest in the model input space. Global (or
probabilistic) SA considers the entire model input space.

We will perform a global variance-based SA, the result of which is a Sobol' index for each of our three inputs.

What is a Sobol' index?
-----------------------
A Sobol' index approximates an exact sensitivity index that results from a global variance-based sensitivity analysis
. Sobol' indices are more efficient to compute than exact sensitivity indices. The magnitude of a (normalized) Sobol'
index indicates how much of the variance in an output quantity of interest can be attributed to variance in a
particular input.

There are different orders of Sobol' index.

* | A **first-order Sobol' index** indicates how much of the variance in the output can be attributed to variance in one
  | particular input alone.
* | A **total-order Sobol' index** indicates how much of the output's variance can be attributed to variance in one
  | particular input, including all of that input's interactions with other inputs.
* | Sobol' indices of orders between 1 and the total number of inputs indicate how much of the output's variance can
  | be attributed to variance in one input, including its interactions of the specified order with other inputs.

We will focus on total-order Sobol' indices for this example.

What is an interaction?
-----------------------
An interaction is that part of the response of an output *f(x)* to the values of x\ :sub:`a`, x\ :sub:`b` "that
cannot be expressed as a superposition of effects separately due to" x\ :sub:`a`, x\ :sub:`b` (Saltelli et al. 2004).

Why might I care about quantifying a component's importance?
============================================================

There are four types of reasons we might want to know how influential an input is with respect to an output (Saltelli
 et al. 2004):

* | **factors prioritisation** -- by setting which factor (x\ :sub:`1`, x \:sub:`2`, or x\ :sub:`3`) to a chosen value
  | will we reduce the uncertainty in *y* the most? We might ask this question if we'd like to rank the inputs by their
  | importance in terms of influencing the output.
* | **factors fixing** -- which factors are not influential in terms of the output *y*? We might ask this question if
  | we wanted to fix some factors without affecting the variance in *y*, rather than letting all of the inputs vary over
  | their domains.
* | **variance cutting** -- if we wanted to achieve a particular reduction in the variance of *y*, which minimal subset
  | of the inputs would we need to fix to particular values to do so? We might ask such a question if we had a limited
  | budget with which to reduce variance in the inputs and wanted to get the most value for our money.
* | **factors mapping** -- which factors are responsible for realisations of *y* in a particular region of interest? We
  | might ask such a question if we were concerned with the safety of the system under investigation, for instance, and
  | wanted to understand what input values and combinations of input values lead to safe vs. dangerous performance.

The sample data set addresses a question in the factors prioritisation setting -- by reducing the fragilities of
which bridges could we improve the expected road network performance the most?

Where can I learn more about sensitivity analysis and/or Sobol' indices?
========================================================================

References
==========
Gitanjali Bhattacharjee and Jack W. Baker. (TBD) *Using global variance-based sensitivity analysis to prioritise
bridge retrofits in a regional road network subject to seismic hazard*. Manuscript under review.

Andrea Saltelli, Stefano Tarantola, Francesca Campolongo, and Marco Ratto. (2004) *Sensitivity Analysis in Practice: A
Guide to Assessing Scientific Models*. John Wiley & Sons, Ltd.

Andrea Saltelli, Marco Ratto, Terry Andres, Francesca Campolongo, Jessica Cariboni, Debora Gatelli, Michaela Saisana,
Stefano Tarantola. (2008) *Global Sensitivity Analysis: The Primer.* John Wiley & Sons, Ltd.