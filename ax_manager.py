import os
import json
from ax.api.client import Client
# UPDATED IMPORTS:
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
from ax.core.parameter import ParameterType, RangeParameter, ChoiceParameter
from ax.core.search_space import SearchSpace
from benchmarks import (ackley,rosenbrock,rastrigin,sphere,beale)

BENCHMARK_REGISTRY = {
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "sphere": sphere,
    "beale": beale
}
EXPERIMENT_DIR = "ax_experiments"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

class AxStateManager:
    def _get_filepath(self, name: str) -> str:
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
        return os.path.join(EXPERIMENT_DIR, f"{clean_name}.json")

    def create_experiment(self, name: str, objective:str,parameters:dict[str, list[float]],maximize: bool = True):
        path = self._get_filepath(name)
        if os.path.exists(path):
            raise ValueError(f"Experiment '{name}' already exists.")
        
        client = Client()
        parameterlist = []
        param_names = parameters.keys()
        for k in param_names:
            parameterlist.append(RangeParameterConfig(name=k,parameter_type='float',bounds=(float(parameters[k][0]),float(parameters[k][1]))))
        # Step 1: Configure Experiment
        client.configure_experiment(
            name=name,
            parameters=parameterlist
        )
        if maximize:
            optval='+'
        else:
            optval='-'
        # Step 2: Configure Optimization
        client.configure_optimization(
            objective=f'{optval}{objective}'
        )

        trials = client.get_next_trials(max_trials=1)
        for trial_index, parameters in trials.items():
            x = parameters["x"]
            y = parameters["y"]
            result = BENCHMARK_REGISTRY[objective]({"x":x,"y":y})
            raw_data = {objective: result}
            client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        best_parameters, prediction, index, name = client.get_best_parameterization()
        client.save_to_json_file(path)
        return f"Created experiment '{name}' at {path}, best {best_parameters}, pred {prediction}"

    def load_client(self, name: str) -> Client:
        path = self._get_filepath(name)
        if not os.path.exists(path):
            raise ValueError(f"Experiment '{name}' not found.")
        client = Client.load_from_json_file(path) 
        return client

    def save_client(self, name: str, client: Client):
        path = self._get_filepath(name)
        client.save_to_json_file(path)
    
    # def add_parameter_to_client(self, name:str, client: Client, param_name: str, bounds: list, param_type: str = "range"):
    #     path = self._get_filepath(name)
    #     if len(client.experiment.trials) > 0:
    #         raise ValueError("Cannot add parameters after trials have started.")

    #     # 1. Clone existing parameters (removing the dummy)
    #     current_params = [
    #         p.clone() for p in client.experiment.search_space.parameters.values() 
    #         if p.name != "init_dummy"
    #     ]
        
    #     # 2. Define the new parameter using Core classes
    #     if param_type == "range":
    #         new_p = RangeParameterConfig(
    #             name=param_name, 
    #             parameter_type='float',
    #             bounds=(float(bounds[0]),float(bounds[1]))
    #         )
    #     else:
    #         raise ValueError("Only 'range' supported for now")
            
    #     current_params.append(new_p)
    #     client.configure_experiment(parameters=current_params)
    #     # 3. Rebuild the search space
    #     # new_search_space = SearchSpace(parameters=current_params)
    #     # client.experiment.search_space = new_search_space
    #     client.save_to_json_file(path)
        
    #     return f"Added {param_name}"