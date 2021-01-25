.. |br| raw:: html

=================================================
Identify important components in complex networks
=================================================

What is **sobol**?
==================

**sobol** is a repository that contains:

* A set of methods for quantifying the importance of individual components in complex networks
* A sample data set on which to use those methods

The sample data set is used to answer the question *Which bridges are the most important to
retrofit if we want to reduce the expected cost of the road network performance?*

It conmprises a set of vector inputs (bridge fragility function parameters) and scalar outputs (the expected cost of
road network performance over a set of earthquake rupture scenarios).

How do we quantify a component's importance?
--------------------------------------------

By estimating its total-order Sobol' index. A Sobol' index approximates an exact sensitivity index that results from a
global variance-based sensitivity analysis. The magnitude of a (normalized) Sobol' index indicates how much of the
variance in an output quantity of interest can be attributed to variance in a particular input.

In the sample data set included in **sobol**, the input quantities describe the fragilities of bridges in the San
Francisco Bay Area road network and the output is a measure of the road network's performance. The bridges'
total-order Sobol' indices tell us how much the fragility of each bridge influences the road network's performance.

Where can I learn more about Sobol' indices or sensitivity analysis?
--------------------------------------------------------------------

There are different types of Sobol' indices and particular conditions under which they can be estimated -- for
details, see the paper associated with this repository or *Global Sensitivity Analysis: The Primer* by Andrea
Saltelli et al.

How can I use **sobol**?
========================

Say we have a physical system whose performance, *y*, depends on three independent variables, x\ :sub:`1`, x\
:sub:`2`, x\ :sub:`3` that take values uniformly at random in the interval [*-\pi*, *\pi*]. The performance of the
system can be described as y = sin x\ :sub:`1` + a sin \ :sup:`2` x\ :sub:`2` + b x\ :sub:`3` :sup:`4` sin x\
:sub:`1`. We would like to know which variable influences the performance of the physical system the most.

There are four types of reasons we might want to know this (Saltelli et al. 2004):

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



How can I install **sobol**?
============================

In Terminal (on Mac), use the following, replacing [myDirectory] with your preferred directory, e.g. ``cd Desktop``.

| ``cd [myDirectory]``
| ``git clone https://github.com/gbhattacharjee/sobol.git``

| Before running any scripts in **sobol**, make sure you have all the required ``Python`` packages in config. Some
packages require specific (earlier) versions, so I recommend creating a virtual environment (e.g., ``venv``) in which
to install them. For example:

| ``cd sobol``
| ``python2 -m venv sobol-venv``
| ``source sobol-venv/bin/activate``
| ``pip install networkx==1.8.1``
| ``pip install pp==1.6.5``
| ``pip install scipy``
| ``pip install numpy``
| ``pip install matplotlib``
| ``pip install plotly``

| To verify installation has worked, please compare the results of the specified function calls to those given in the
 benchmarks document.

References
==========
Andrea Saltelli, Stefano Tarantola, F. Campolongo, and M. Ratto. (2004) *Sensitivity Analysis in Practice: A Guide to
Assessing Scientific Models*. John Wiley & Sons, Ltd.