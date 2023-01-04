# On Legible and Predictable Robot Navigation in Multi-Agent Environments

## Abstract
Legible motion is intent-expressive, which when employed during social robot navigation, allows others to quickly infer the intended avoidance strategy. Predictable motion matches an observer’s expectation which, during navigation, allows others to confidently carryout the interaction. In this work, we present a navigation framework capable of reasoning on its legibility and predictability with respect to dynamic interactions, e.g., a passing side. Our approach generalizes the previously formalized notions of legibility and predictability by allowing dynamic goal regions in order to navigate in dynamic environments. This generalization also allows us to quantitatively evaluate the legibility and the predictability of trajectories with respect to navigation interactions. Our approach is shown to promote legible behavior in ambiguous scenarios and predictable behavior in unambiguous scenarios. In a multi-agent environment, this yields an increase in safety while remaining competitive in terms of goal-efficiency when compared to other robot navigation planners in multi-agent environments.

## Installation
1. Clone the repository with its submodules and install it locally by running
```shell
git clone --recursive https://github.com/jlbas/LPSNav.git
cd LPSNav
pip install .
```
2. Install the [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
```shell
cd lpsnav/policies/Python-RVO2
pip install cython
python setup.py build
python setup.py install
```

## Usage
Run a simulation from the project's root folder
```shell
python sim.py
```

## Configuration
The simulation parameters are configured in `config/config.toml`

### Adding your own policy
A new policy, `Your-New-Policy`, can be added to the simulation environment by doing the following:
1. Add a `your_new_policy` subsection to the `agent` section of the `config.toml` file, along with any required parameters
```toml
[agent]

[agent.your_new_policy]
name = "Your-New-Policy"
req_param_1 = "some_value"
req_param_2 = "another_value"
```
2. Create the module `policies/your_new_policy.py` and implement a `YourNewPolicy` class which inherits from `Agent`
```python
from policies.agent import Agent

class YourNewPolicy(Agent):
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, rng):
        super().__init__(conf, id, policy, is_ego, max_speed, start, goal, rng)
        self.req_param_1 = conf["req_param_1"]
        self.req_param_2 = conf["req_param_2"]
```
3. Create a `get_action` method to set the desired speed and heading
```python
...
class  YourNewPolicy(Agent):
  ...
  def get_action(self, dt, agents):
    self.des_speed = 1
    self.des_heading = 0
```
4. Lastly, add `"your_new_policy"` to the list of policies to be simulated in the `scenario` section of the `config/config.toml` file
```toml
[scenario]
policy=["your_new_policy"]
```
