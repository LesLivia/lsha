L*_sha: L*-Based Algorithm for Stochastic Hybrid Automata Learning 
====================================

This repository contains the implementation of L*_sha, an algorithm to learn Stochastic Hybrid Automata from collected system traces.
The algorithm is not tied to a specific domain, although it has been successfully exploited
to infer a model of human behavior in human-robot interaction scenarios building upon the work presented in:
- [*Formal Modeling and Verification of Multi-Robot Interactive Scenarios in Service Settings*][paper5]
- [*A Deployment Framework for Formally Verified Human-Robot Interactions*][paper4]
- [*A Model-driven Approach for the Formal Analysis of Human-Robot Interaction Scenarios*][paper3]
- [*Formal Verification of Human-Robot Interaction in Healthcare Scenarios*][paper2]
- [*Statistical Model Checking of Human-Robot Interaction Scenarios*][paper1]

The algorithm builds upon [L\*][angluin], whose main elements are:
- a [Teacher](sha_learning/learning_setup/teacher.py) that stores the collected traces and answers queries based on currently accumulated knowledge 
- a [Learner](sha_learning/learning_setup/learner.py) that progressively refines the hypothesis automaton by asking queries to the Teacher

The teacher relies on **samples** of the System Under Learning (SUL) (i.e., it does not possess exact knowledge).
To generate traces for the specific use case of human-robot interaction, we exploit either [Uppaal][uppaal] with manually drafted SHA to be learned or the [deployment framework][dep] to simulate the robotic application in a virtual environment and collect the simulation traces.

Authors:

| Name              | E-mail address           |
|:----------------- |:-------------------------|
| Lestingi Livia    | livia.lestingi@polimi.it |

Learned SHA
-----------

Learned SHA for the **thermostat** case study can be found [here](resources/learned_ha/thermostat_cs).

Learned SHA for the **human-robot interaction** case study can be found [here](resources/learned_ha/hri_cs).

Configuration File Setup
-----------

The [main L\*_sha script](sha_learning/learn_model.py) requires as input parameter the path to a configuration file, whose template can be found within the [`./resources/config/`](sha_learning/resources/config) folder.

Make sure to set each property to match your environment, specifically: 
- **N_min** is the minimum number of observations for each trace to stop perfoming the refinement query (i.e., a value greater than 10 is advised);
- **CASE_STUDY** is the chosen SUL (either THERMO or HRI);
- **CS_VERSION** is the experiment you want to perform for the chosen SUL;
- **RESAMPLE_STRATEGY** is the chosen approach to generate new SUL traces (either UPPAAL or SIM).

If the chosen resample strategy is UPPAAL:
- **UPPAAL_PATH** is the path to Uppaal [command line utility][verifyta];
- **UPPAAL_SCRIPT_PATH** is the path to [*verify.sh*](resources/scripts);
- **UPPAAL_MODEL_PATH** is the path to the Uppaal model template (e.g., [*hri-w_ref.xml*](resources/uppaal_resources/hri-w_ref.xml)); 
- **UPPAAL_QUERY_PATH** is the path to the Uppal query template (e.g., [*thermostat.q*](resources/uppaal_resources/thermostat.q));
- **UPPAAL_OUT_PATH** is the path where you want the generated traces to be stored.

If the chosen resample strategy is SIM:
- **SIM_LOGS_PATH** is the path to available SUL traces.

**Note**: The algorithm has been tested on Uppaal **v.4.1.24** on Mac OS X. Should you run into any issue while testing with a different configuration please report to livia.lestingi@polimi.it.

Python Dependencies
-----------

Install the required dependencies:

	pip install -r $LSHA_REPO_PATH/requirements.txt

Add the L\*\_SHA repo path to your Pytho path (fixes ModuleNotFoundError while trying to execute from command line):

	export PYTHONPATH="${PYTHONPATH}:$LSHA_REPO_PATH"

Run the main script specifying the path to your configuration file:

	python3 $LSHA_REPO_PATH/it/polimi/hri_learn/learn_model.py $CONFIG_FILE_PATH
	
---

*Copyright &copy; 2021 Livia Lestingi*

[paper1]: https://doi.org/10.4204/EPTCS.319.2
[paper2]: https://doi.org/10.1007/978-3-030-58768-0_17
[paper3]: https://doi.org/10.1109/SMC42975.2020.9283204
[paper4]: https://doi.org/10.1109/ACCESS.2021.3117852
[paper5]: https://ieeexplore.ieee.org/abstract/document/9796464 
[angluin]: https://doi.org/10.1016/0890-5401(87)90052-6
[uppaal]: https://uppaal.org/
[dep]: https://github.com/LesLivia/hri_deployment
[verifyta]: https://docs.uppaal.org/toolsandapi/verifyta/