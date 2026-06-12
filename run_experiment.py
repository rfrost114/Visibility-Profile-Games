import os
from datetime import datetime
import logging
import sys
from PSRO import PSRO

def setup_logging(base_dir='./results'):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}_200node_patroldef")

    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_dir, 'training_execution_log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Execution Directory Initialized at: {run_dir}")

    return run_dir

if __name__ == '__main__':
    active_run_dir = setup_logging()

    test = PSRO(
        graph_config_name='example_2v2.yml',
        defender_policy_name='Patrolling_Def',
        utility_expectation_horizon=20
    )

    # training_args = {
    #     'timesteps' : 2_500_000,
    #     'clip_range' : 0.2,
    #     'target_kl' : 0.05,
    #     'ent_coef' : 0.05,
    #     'num_envs' : 4,
    #     'n_steps' : 250,
    #     'tensorboard_log': os.path.join(active_run_dir, "tb_logs"),
    #     'save_path': os.path.join(active_run_dir, "checkpoints")
    # }
    training_args = {
        'tensorboard_log': os.path.join(active_run_dir, "tb_logs"),
        'save_path': os.path.join(active_run_dir, "checkpoints"),
        'timesteps' : 1_000_000
    }
    test.solve(training_args, run_dir=active_run_dir)