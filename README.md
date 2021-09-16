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

ROS Package
-----------

The standalone Python robot controller and the robot and humans in the simulated scene 
communicate over ROS.
Installing [ROS Melodic][ros] is necessary to proceed.

The custom package has to be built within the catkin workspace (paths should be properly modified):

	cd $REPO_PATH/catkin_ws
	catkin_make

Source the generated setup file:

	source $REPO_PATH/catkin_ws/devel/setup.bash

V-Rep Scene
-----------

Before running V-Rep, make sure `roscore` is running and the [ROS Interface][rosint]
is correctly loaded.

The custom scene can be opened either via the GUI or the following command:

	./$VREP_PATH/vrep.sh $REPO_PATH/hri_deployment/VRep_Scenes/hri_healthcare_scene.ttt

**Note**: The human is controlled through the following keyboard inputs:

| Key            | Action                |
|:---------------|:----------------------|
| Up Arrow       | Walk                  |
| Down Arrow     | Stop                  |
| Left Arrow     | Turn Left             |
| Right Arrow    | Turn Right            |
| Tab            | Switch to Other Human |
| Enter          | Set as Served         |


Python Dependencies
-----------

Make sure you have the required dependencies installed:

	pip install -r $REPO_PATH/py_scripts/requirements.txt

Finally, run the controller Python script:

	python3 $REPO_PATH/py_scripts/main.py
	
---

*Copyright &copy; 2021 Livia Lestingi*

[angluin]: https://doi.org/10.1016/0890-5401(87)90052-6
[vrep]: https://coppeliarobotics.com/downloads
[ros]: http://wiki.ros.org/melodic/Installation
[rosint]: https://www.coppeliarobotics.com/helpFiles/en/rosInterf.htm
