L*_sha: L*-Based Algorithm for Stochastic Hybrid Automata Learning 
====================================

This repository contains the implementation of L\*_sha, an algorithm to learn key features
of Stochastic Hybrid Automata from collected system traces.
The algorithm is not tied to a specific domain, although it has been successfully exploited
to infer a model of human behavior in human-robot interaction scenarios.

The algorithm has the same structure as [L\*][angluin] which consists of:
- a [Teacher](it/polimi/hri_learn/lstar_sha) that stores all the collected runs of the system and is able to answer queries based on currently accumulated knowledge 
- a [Learner](it/polimi/hri_learn/lstar_sha) that maintains and progressively refines the hypothesis automaton modeling the system under learning through answers to queries provided by the Teacher

**Note**: The framework has been tested on Ubuntu 18.04. Should you succeed in running it
on a different OS, please report any issue or interesting result to *livia.lestingi@polimi.it*.

Authors:

| Name              | E-mail address           |
|:----------------- |:-------------------------|
| Lestingi Livia    | livia.lestingi@polimi.it |


Python Dependencies
-----------

Make sure you have the required dependencies installed:

	pip install -r $REPO_PATH/requirements.txt

Finally, run the controller Python script:

	python3 $REPO_PATH/it/polimi/hri_learn/refine_model.py
	
---

*Copyright &copy; 2021 Livia Lestingi*

[angluin]: https://doi.org/10.1016/0890-5401(87)90052-6
[vrep]: https://coppeliarobotics.com/downloads
[ros]: http://wiki.ros.org/melodic/Installation
[rosint]: https://www.coppeliarobotics.com/helpFiles/en/rosInterf.htm