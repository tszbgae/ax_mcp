# server.py
from mcp.server.fastmcp import FastMCP
from ax_manager import AxStateManager
import benchmarks
from benchmarks import (ackley,rosenbrock,rastrigin,sphere,beale)

BENCHMARK_REGISTRY = {
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "sphere": sphere,
    "beale": beale
}

# Initialize Server
mcp = FastMCP("AxOptimizationAgent")
manager = AxStateManager()

@mcp.tool()
def create_study(study_name: str, objective_name: str, parameters:dict[str, list[float]], maximize: bool = True) -> str:
    """
    Initialize a new Ax optimization study.
    Args:
        study_name: Unique identifier for this experiment.
        objective_name: What we are measuring (e.g., 'accuracy', 'latency').
        maximize: True if we want the metric to go up, False for down.
    """
    try:
        # We create with a dummy parameter because Ax requires at least one.
        # The LLM will overwrite/add real ones next.
        return manager.create_experiment(study_name, objective_name,parameters,maximize)
    except Exception as e:
        return f"Error: {str(e)}"

# @mcp.tool()
# def add_parameter(study_name: str, param_name: str, param_type: str, bounds: list[float], value_type: str = "float") -> str:
#     """Add a search parameter to the study."""
#     try:
#         client = manager.load_client(study_name)
        
#         # Use our new helper method in the manager
#         manager.add_parameter_to_client(client, param_name, bounds, param_type)
        
#         manager.save_client(study_name, client)
#         return f"Parameter {param_name} added to {study_name}"
#     except Exception as e:
#         return f"Error adding parameter: {str(e)}"

@mcp.tool()
def get_and_complete_next_trial(study_name: str,objective_name:str,number_loops:int) -> str:
    """
    Generates the next set of parameters to test.
    Completes a loop of number_loops times.
    Then completes the trials number_loops times.
    Returns a note for success completion.
    """
    client = manager.load_client(study_name)
    try:
        trials = client.get_next_trials(max_trials=number_loops)
        for trial_index, parameters in trials.items():
            x = parameters["x"]
            y = parameters["y"]
            result = BENCHMARK_REGISTRY[objective_name]({"x":x,"y":y})
            raw_data = {objective_name: result}
            client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        manager.save_client(study_name, client) # Save state (trial is now 'running')
        return 'success'
    except Exception as e:
        return f"Error generating trial: {str(e)}"

# @mcp.tool()
# def complete_trial(study_name: str, trial_index: int, metric_value: float) -> str:
#     """
#     Report the results of a trial back to Ax.
#     """
#     client = manager.load_client(study_name)
#     try:
#         client.complete_trial(trial_index=trial_index, raw_data=metric_value)
#         manager.save_client(study_name, client)
#         return f"Trial {trial_index} completed with value {metric_value}."
#     except Exception as e:
#         return f"Error completing trial: {str(e)}"

@mcp.tool()
def list_available_functions() -> str:
    """
    Returns a list of available optimization benchmark functions 
    and their descriptions/recommended bounds.
    """
    return str(benchmarks.get_function_info())

@mcp.tool()
def evaluate_benchmark(function_name: str, parameters: dict[str, float]) -> str:
    """
    Calculates the value of a benchmark function for specific parameters.
    Use this outside of an Ax study to show benchmark functionality.  
    Args:
        function_name: Must be one of 'ackley', 'rosenbrock', 'rastrigin', 'sphere', 'beale'.
        parameters: A dictionary of floats, e.g. {"x": 1.5, "y": -0.5}.
    """
    try:
        result = benchmarks.evaluate(function_name, parameters)
        return f"Function '{function_name}' result: {result}"
    except Exception as e:
        return f"Error evaluating function: {str(e)}"

@mcp.tool()
def provide_best_parameters(study_name:str) -> str:
    """
    From the trials of the active ax study_name, provides the best parameters 
    and the prediction of the objective function with these parameters
    """
    client = manager.load_client(study_name)
    best_parameters, prediction, index, name = client.get_best_parameterization()
    outstring = f''
    for x in best_parameters.keys():
        outstring = outstring + f'{x}: {best_parameters[x]}'
    pk = list(prediction.keys())[0]
    outstring = outstring + f'. {prediction[pk][0]}, {prediction[pk][1]}'
    print(outstring)
    return outstring


# @mcp.tool()
# def run_benchmark_trial(study_name: str, function_name: str) -> str:
#     """
#     Executes a full optimization step on the server side:
#     1. Generates parameters from Ax.
#     2. Runs the specified benchmark function immediately.
#     3. Reports the result back to Ax.
    
#     Args:
#         study_name: The experiment name.
#         function_name: The benchmark to test (e.g., 'rosenbrock', 'ackley').
#     """
#     client = manager.load_client(study_name)
    
#     try:
#         # 1. Ask Ax for the next parameters
#         param_dict, trial_index = client.get_next_trial()
        
#         # 2. Run the function (The "Execution" Step)
#         # This is where the function is selected and executed!
#         try:
#             result_value = benchmarks.evaluate(function_name, param_dict)
#         except ValueError as ve:
#             # If function not found, mark trial failed so Ax knows not to rely on it
#             client.log_trial_failure(trial_index=trial_index)
#             return f"Error: {str(ve)}. Trial {trial_index} marked as failed."

#         # 3. Report back to Ax
#         client.complete_trial(trial_index=trial_index, raw_data=result_value)
#         manager.save_client(study_name, client)
        
#         return (f"Trial {trial_index} COMPLETE.\n"
#                 f"Params: {param_dict}\n"
#                 f"Function: {function_name}\n"
#                 f"Result: {result_value}")

#     except Exception as e:
#         return f"Optimization loop failed: {str(e)}"

if __name__ == "__main__":
    mcp.run()