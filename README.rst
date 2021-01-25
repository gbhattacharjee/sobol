.. |br| raw:: html

=================================================
Identify important components in complex networks
=================================================

What is **sobol**?
==================

**sobol** is a repository that contains:

* A set of methods for quantifying the importance of individual components in complex networks
* An example data set on which to use those methods -- a set of inputs (bridge fragility function parameters) and a
vector of outputs (the expected cost of road network performance over a set of earthquake rupture scenarios).

One of the questions we can answer with this data set and these methods is *Which bridges are the most important to
retrofit if we want to reduce the expected cost of the road network performance?*

How do we quantify a component's importance?
--------------------------------------------

By estimating its total-order Sobol' index. A Sobol' index approximates a sensitivity index that results from a
global variance-based sensitivity analysis. The magnitude of a (normalized) Sobol' index indicates how much of the
variance in an output quantity of interest can be attributed to variance in a particular input.

In the data set included in **sobol**, the input quantities describe the fragilities of bridges in the San Francisco
Bay Area road network and the output is a measure of the road network's performance. The bridges' total-order Sobol'
indices tell us how much the fragility of each bridge influences the road network's performance.

Where can I learn more about Sobol' indices or sensitivity analysis?
--------------------------------------------------------------------

There are different types of Sobol' indices and particular conditions
under which they can be estimated -- for details, see the paper associated with this repository or *Global
Sensitivity Analysis: The Primer* by Andrea Saltelli et al.

How can I use **sobol**?
========================

Say we have a physical system whose performance, *y*, depends on three variables, *x\ :sub:`1`*, *x\ :sub:`2`*, *x\
:sub:`3`* that take values uniformly at random in the interval [*-\pi*, *\pi*].

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

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


