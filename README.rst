=================================================
Identify important components in complex networks
=================================================

What is **sobol**?
==================

The name **sobol** refers to Sobol' indices, which are approximations of the sensitivity indices that result from global
variance-based sensitivity analysis. A Sobol' index quantifies how much of the variance in an output quantity of interest
can be attributed to variance in a particular input. There are different types of Sobol' indices and particular conditions
under which they can be estimated -- for details, see the paper associated with this repository or *Global
Sensitivity Analysis: The Primer* by Andrea Saltelli et al.

This repository contains:

* A set of methods for quantifying the importance of individual components in complex networks
* A data set on which to use those methods -- a set of inputs (bridge fragility function parameters) and a vector of
outputs (the expected cost of road network performance over a set of earthquake rupture scenarios).

One of the questions we can answer with this data set and these methods is *Which bridges are the most important to
retrofit if we want to reduce the expected cost of the road network performance?*

How can I install **sobol**?
============================

In Terminal (on Mac):

|``cd [myDirectory]``
|``git clone https://github.com/gbhattacharjee/sobol.git``
