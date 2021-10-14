L*_sha: L*-Based Algorithm for Stochastic Hybrid Automata Learning 
====================================

This repository contains the implementation of L\*_sha, an algorithm to learn key features
of Stochastic Hybrid Automata from collected system traces.
The algorithm is not tied to a specific domain, although it has been successfully exploited
to infer a model of human behavior in human-robot interaction scenarios building upon the work presented in:
- [*A Deployment Framework for Formally Verified Human-Robot Interactions*][paper4]
- [*A Model-driven Approach for the Formal Analysis of Human-Robot Interaction Scenarios*][paper3]
- [*Formal Verification of Human-Robot Interaction in Healthcare Scenarios*][paper2]
- [*Statistical Model Checking of Human-Robot Interaction Scenarios*][paper1]

The algorithm has the same structure as [L\*][angluin] which consists of:
- a [Teacher](it/polimi/hri_learn/lstar_sha) that stores the collected traces and answers queries based on currently accumulated knowledge 
- a [Learner](it/polimi/hri_learn/lstar_sha) that progressively refines the hypothesis automaton by asking queries to the Teacher

The L\*_sha Teacher relies on **samples** of the System Under Learning (SUL).
To generate traces for the specific use case of human-robot interaction, we exploit either [Uppaal][uppaal] with manually drafted SHA to be learned or the [deployment framework][dep] to simulate the robotic application in a virtual environment and collect the simulation traces.

Authors:

| Name              | E-mail address           |
|:----------------- |:-------------------------|
| Lestingi Livia    | livia.lestingi@polimi.it |

Configuration File Setup
-----------

The main L\*_sha




Python Dependencies
-----------

Make sure you have the required dependencies installed:

	pip install -r $REPO_PATH/requirements.txt

Finally, run the controller Python script:

	python3 $REPO_PATH/it/polimi/hri_learn/learn_model.py $CONFIG_FILE_PATH
	
---

*Copyright &copy; 2021 Livia Lestingi*

[paper1]: https://doi.org/10.4204/EPTCS.319.2
[paper2]: https://doi.org/10.1007/978-3-030-58768-0_17
[paper3]: https://doi.org/10.1109/SMC42975.2020.9283204
[paper4]: https://doi.org/10.1109/ACCESS.2021.3117852
[angluin]: https://doi.org/10.1016/0890-5401(87)90052-6
[uppaal]: https://uppaal.org/
[dep]: https://github.com/LesLivia/hri_deployment