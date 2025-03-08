"""
MaAS Optimization Script

This script provides a command-line interface for running optimization experiments using the 
Multi-agent Architecture Search (MaAS) framework. MaAS dynamically samples customized multi-agent 
systems for each query, balancing performance and computational cost.

The optimization workflow:
1. Loads experiment configuration based on the selected dataset
2. Configures LLM models for optimization and execution tasks
3. Initializes the Optimizer with the specified parameters
4. Runs the optimization process in either Test or Graph mode

The Optimizer performs cost-aware empirical Bayes Monte Carlo optimization, updating agentic 
operators using textual gradient-based methods when enabled.
"""

import argparse
from maas.configs.models_config import ModelsConfig
from maas.ext.maas.scripts.optimizer import Optimizer
from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS

def parse_args():
    parser = argparse.ArgumentParser(description="MAAS Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset type to use for optimization. Must be one of the predefined experiment configurations.",
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=4, 
        help="Number of samples to use during optimization. Higher values provide more robust results but increase computation time."
    )
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="maas/ext/maas/scripts/optimized",
        help="Directory path where optimized results will be saved for later analysis or use.",
    )
    parser.add_argument(
        "--round", 
        type=int, 
        default=1, 
        help="Specifies which optimization round to run. Multiple rounds can be used for iterative refinement."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size used during training. Affects memory usage and optimization speed."
    )
    parser.add_argument(
        "--opt_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Name of the LLM model used for optimization tasks. Must be defined in the models configuration.",
    )
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Name of the LLM model used for execution tasks. Must be defined in the models configuration.",
    )
    parser.add_argument(
        "--is_test",
        type=bool, 
        default=False, 
        help="When True, runs the optimizer in Test mode instead of Graph mode. Test mode is useful for debugging."
    )
    parser.add_argument(
        "--is_textgrad", 
        type=bool, 
        default=False, 
        help="When True, enables textual gradient-based optimization methods for updating agentic operators."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        help="Learning rate for the optimization process. Controls the step size during updates."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load the experiment configuration for the selected dataset
    config = EXPERIMENT_CONFIGS[args.dataset]

    # Load model configurations from the default configuration file
    models_config = ModelsConfig.default()
    
    # Get the optimization model configuration
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{args.opt_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --opt_model_name flag. "
        )

    # Get the execution model configuration
    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --exec_model_name flag. "
        )

    # Initialize the Optimizer with the specified parameters
    optimizer = Optimizer(
        dataset=config.dataset,          # Dataset from the experiment configuration
        question_type=config.question_type,  # Type of questions in the dataset
        opt_llm_config=opt_llm_config,   # LLM configuration for optimization
        exec_llm_config=exec_llm_config, # LLM configuration for execution
        operators=config.operators,      # Agentic operators from the experiment configuration
        optimized_path=args.optimized_path,  # Path to save optimization results
        sample=args.sample,              # Number of samples for optimization
        round=args.round,                # Optimization round
        batch_size=args.batch_size,      # Batch size for training
        lr=args.lr,                      # Learning rate
        is_textgrad=args.is_textgrad,    # Whether to use textual gradient-based methods
    )

    # Run the optimization process in either Test or Graph mode
    if args.is_test:
        optimizer.optimize("Test")      # Test mode for debugging
    else:
        optimizer.optimize("Graph")     # Graph mode for full optimization

"""
Usage Examples:

1. Basic optimization on GSM8K dataset:
   python examples/maas/optimize.py --dataset gsm8k

2. Optimization with increased samples for better results:
   python examples/maas/optimize.py --dataset math --sample 8 --batch_size 8

3. Using textual gradient-based optimization:
   python examples/maas/optimize.py --dataset humaneval --is_textgrad True --lr 0.005

4. Running in test mode for debugging:
   python examples/maas/optimize.py --dataset gaia --is_test True

5. Using specific models for optimization and execution:
   python examples/maas/optimize.py --dataset multiarith --opt_model_name gpt-4 --exec_model_name claude-3-opus
"""
