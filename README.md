# On Legible and Predictable Robot Navigation in Multi-Agent Environments

## Installation
1. Clone the repository and install it locally by running
```shell
git clone https://github.com/jlbas/LPSNav.git
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
...
[agent]
...
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
4. Lastly, add `"your_new_policy"` to the list of policies to be simulated in the `scenario` section of the `config/confi.toml` file
```toml
[scenario]
...
policy=["your_new_policy"]
```
