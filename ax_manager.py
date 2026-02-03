import os
import json
from ax.api.client import Client
# UPDATED IMPORTS:
from ax.core.parameter import ParameterType, RangeParameter, ChoiceParameter
from ax.core.search_space import SearchSpace

EXPERIMENT_DIR = "ax_experiments"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

class AxStateManager:
    def _get_filepath(self, name: str) -> str:
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
        return os.path.join(EXPERIMENT_DIR, f"{clean_name}.json")

    def create_experiment(self, name: str, maximize: bool = True):
        path = self._get_filepath(name)
        if os.path.exists(path):
            raise ValueError(f"Experiment '{name}' already exists.")
        
        client = Client()
        
        # Step 1: Configure Experiment
        client.configure_experiment(
            experiment_name=name,
            parameters=[
                {
                    "name": "init_dummy",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float"
                }
            ]
        )
        
        # Step 2: Configure Optimization
        client.configure_optimization(
            objective_name="objective",
            minimize=not maximize,
        )

        client.save_to_json_file(path)
        return f"Created experiment '{name}' at {path}"

    def load_client(self, name: str) -> Client:
        path = self._get_filepath(name)
        if not os.path.exists(path):
            raise ValueError(f"Experiment '{name}' not found.")
        return Client.load_from_json_file(path)

    def save_client(self, name: str, client: Client):
        path = self._get_filepath(name)
        client.save_to_json_file(path)

    def add_parameter_to_client(self, client: Client, param_name: str, bounds: list, param_type: str = "range"):
        if len(client.experiment.trials) > 0:
            raise ValueError("Cannot add parameters after trials have started.")

        # 1. Clone existing parameters (removing the dummy)
        current_params = [
            p.clone() for p in client.experiment.search_space.parameters.values() 
            if p.name != "init_dummy"
        ]
        
        # 2. Define the new parameter using Core classes
        if param_type == "range":
            new_p = RangeParameter(
                name=param_name, 
                parameter_type=ParameterType.FLOAT, 
                lower=float(bounds[0]), 
                upper=float(bounds[1])
            )
        else:
            raise ValueError("Only 'range' supported for now")
            
        current_params.append(new_p)

        # 3. Rebuild the search space
        new_search_space = SearchSpace(parameters=current_params)
        client.experiment.search_space = new_search_space
        
        return f"Added {param_name}"