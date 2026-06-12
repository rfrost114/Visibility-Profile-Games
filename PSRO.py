import numpy as np
import random
import glob
import os
import json
import time
import cvxpy as cp
import networkx as nx
from Simulator import SimulatorEnv
from PartitionEnv import PartitionEnv
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE, Compose
from SB3Callbacks.RewardAnnealing import RewardAnnealingCallback
from SB3Callbacks.EarlyStoppingCallback import EarlyStoppingCallback
from SB3Callbacks.CheckpointCallback import CheckpointCallback
from SB3Callbacks.LoggingCallback import LoggingCallback
from stable_baselines3.common.callbacks import CallbackList
from networks.MaskableGNNPolicy import MaskableGNNPolicy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import logging
import sys


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )

class PSRO:

    def __init__(
            self,
            graph_config_name : str,
            defender_policy_name : str,
            utility_expectation_horizon : int,
            logs : bool = True
    ):
        self.graph_config_name  = graph_config_name
        self.defender_policy_name = defender_policy_name
        self.utility_expectation_horizon = utility_expectation_horizon
        self.logging = logs

        # build attacker policy dict
        # open attacker policy folder, get all file names, strip .py, index
        self.attacker_policies = self._create_attacker_policy_dict()
        
        # initialize the simulator used for policy evaluation
        self.simulator = SimulatorEnv(self.graph_config_name, self.defender_policy_name)
        self.num_nodes = self.simulator.num_nodes
        self.num_defenders = self.simulator.num_defenders
        self.edge_index, self.lap_pos, self.walk_pos = self._get_graph_info()

        if self.logging:
            logging.info('----- Problem Params -----')
            logging.info(f'Attacker Policies: {len(self.attacker_policies)}')
            logging.info(f'Defense Policy: {self.defender_policy_name}')
            logging.info(f'Number Defenders: {self.num_defenders}')
            logging.info(f'Config Name: {self.graph_config_name}')
            logging.info(f'Graph Size : {self.num_nodes} nodes')

        # inital defense policy is not top use an information partition
        self.partition_policies = {0 : [list(range(self.num_nodes)) for _ in range(self.num_defenders)]}


        # holds lists of policy ids  
        inital_atk_policy = random.choice(list(self.attacker_policies.keys()))
        # inital_atk_policy = 2 #DELETE THIS!!!
        if self.logging:
            logging.info(f'Attacker seed policy: {self.attacker_policies[inital_atk_policy]}')
        self.subgame_polices = {
            'a' : [inital_atk_policy], # choose the inital policy for the attacker at random
            'd' : [0] # there is only one defender policy initially 
        }

        ### subgame policy vectors
        # mixed policy vector for the attack policies, this might change length over time
        self.attacker_vector = np.ones(1, dtype=np.float32)
        # mixed policy vector for defense, this might change length over time 
        self.partition_vector = np.ones(1, dtype=np.float32)

        # utility matrix is essentailly a look up table 
        self.utility_matrix = self._compute_initial_utility_matrix() # pay off utils for inial policies policies
        if self.logging:
            logging.info(f'Unmodified Game Payoff: {float(np.max(self.utility_matrix))}')
    

    def solve(
            self,
            training_args : dict,
            epsilon : float = 1e-3,
            run_dir: str = "./results"
    ):
        '''
        Training Args
        - num_envs : 8
        - hidden_dim : 64 
        - n_steps: 250
        - batch_size: 256
        - learning_rate: 3e-4
        - target_kl: 0.015
        - clip_range: 0.15
        - ent_coef: 0.01
        - verbose: 0 
        - tensorboard_log: ppo_tensorboard_logs --- must explicitly be set to None to avoid logging
        - save_path : checkpoints 
        - timesteps: 2_500_000 experimentally suitable for 100 node graph 

        Note:
        num_envs * n_steps =2000 seems to work well
        '''

        old_utility = np.inf
        iters = 1
        prev_attacker_included = False

        while True:
            loop_start_time = time.time()
            if self.logging:
                logging.info(f'----- Iteration {iters} -----')

            attacker_included, defender_included = False, False
            ### calculate current subgame util

            subgame_matrix = self._form_subgame_matrix() 
            if iters == 1: # on the first iter, subgame matrix is singleton so we dont need to call the lp
                current_utility = subgame_matrix.item()
            else:
                current_utility = self._calculate_subgame_nash(subgame_matrix)

            if self.logging:
                logging.info(f'Current Subgame Utility: {current_utility:.4f}')
                logging.info(f'Current Atk Subgame Policies: {self.subgame_polices["a"]}')
                logging.info(f'Current Def Subgame Policies: {self.subgame_polices["d"]}')
                logging.info(f'Subgame Attacker Mixed : {self.attacker_vector}')
                logging.info(f'Subgame Defender Mixed : {self.partition_vector}')


            # end if we have not made sufficent progress
            if np.abs(old_utility - current_utility) < epsilon and prev_attacker_included:
                if self.logging:
                    logging.info(f'Terminating! - Lack of Progress')
                break
            
            
            
            attacker_br = self._find_attacker_br()
            attacker_included = attacker_br in self.subgame_polices['a']
            prev_attacker_included = attacker_included


            if self.logging:
                logging.info(f'Attacker BR: Policy {attacker_br} - {self.attacker_policies[attacker_br]}')

            defender_br = self._find_defense_br(training_args, iters)
            defender_included = defender_br in self.partition_policies.values()

            # terminate if both policies are already in the subgame
            if attacker_included and defender_included:
                if self.logging:
                    logging.info(f'Terminating! - No New BR')
                break

            if not attacker_included:
                self.subgame_polices['a'].append(attacker_br)
            
            if not defender_included:
                new_key = max(self.partition_policies.keys())+1
                self.partition_policies[new_key] = defender_br
                self.subgame_polices['d'].append(new_key)
                self._recalculate_utilities(defender_br)


            # if attacker_br not in self.subgame_polices['a']: # check to see if this policy is already part of the subgame
            #     self.subgame_polices['a'].append(attacker_br)
            # else:
            #     attacker_included = True
            # if self.logging:
            #     logging.info(f'Attacker BR: Policy {attacker_br}')
            
            # defender_br = self._find_defense_br(training_args, iters)
            # if defender_br not in self.partition_policies.values(): # see if we have already found this partition (unlikely)
            #     new_key = max(self.partition_policies.keys())+1
            #     self.partition_policies[new_key] = defender_br
            #     self.subgame_polices['d'].append(new_key)
            # else:
            #     defender_included = True
            

            
            # # terminate if both policies are already in the subgame
            # if attacker_included and defender_included:
            #     if self.logging:
            #         logging.info(f'Terminating! - No New BR')
            #     break


            # # update utility matrix
            # if not defender_included: # no need to update if we already have this partition policy
            #     self._recalculate_utilities(defender_br)

            old_utility = current_utility

            loop_duration = time.time() - loop_start_time
            self._log_iter_to_json(iters, current_utility, loop_duration, run_dir)
            iters += 1
        
        # after we exit, recalculate the subgame nash a final time (might be redundant, fix if large instance)
        subgame_matrix = self._form_subgame_matrix()
        final_value = self._calculate_subgame_nash(subgame_matrix)
        
        full_attacker_policy = self._to_full_policy(self.attacker_vector, 'a')
        full_partition_policy = self._to_full_policy(self.partition_vector, 'p')
        self.attacker_vector = full_attacker_policy
        self.partition_vector = full_partition_policy
        if self.logging:
            logging.info(f'Final Game Value {final_value}')
            logging.info(f'Final Attacker Mixed : {self.attacker_vector}')
            logging.info(f'Final Defender Mixed : {self.partition_vector}')

        loop_duration = time.time() - loop_start_time
        self._log_iter_to_json(iters, final_value, loop_duration, run_dir)

        return final_value, self.attacker_vector, self.partition_vector

        
        
    # unpacks a vector policy for the subgame to be compatible with the full utility matrix
    def _to_full_policy(self, sg_vector : np.ndarray, player : str):
        if player.startswith('a'):
            full_policy = np.zeros(len(self.attacker_policies))
            sg_policies = self.subgame_polices['a']

        else:
            full_policy = np.zeros(len(self.partition_policies))
            sg_policies = self.subgame_polices['d']

        for full_index, weight in zip(sg_policies, sg_vector):
            full_policy[full_index] = weight
        return full_policy




    def _create_attacker_policy_dict(self) -> dict[int:str]:
        attacker_policy_paths = [os.path.basename(x)[:-3] for x in glob.glob('policies/attacker/*.py')]
        return {idx : policy_name  for idx, policy_name in enumerate(attacker_policy_paths)}

    
    def _compute_initial_utility_matrix(self):
        utilities = np.zeros((len(self.partition_policies), len(self.attacker_policies)))
        for i in range(utilities.shape[0]): # partitions
            for j in range(utilities.shape[1]): # attackers
                # print(f'{self.partition_policies[i]=}')
                # print(f'{self.attacker_policies[j]=}')
                utilities[i,j] = np.mean([self._simulate(self.partition_policies[i], self.attacker_policies[j]) for _ in range(self.utility_expectation_horizon)])
        return utilities

    def _simulate(self, partition_policy, attack_policy) -> float:
        self.simulator.reset()
        self.simulator.set_attacker_heuristic(attack_policy)
        self.simulator.set_partition(partition_policy)
        captures = self.simulator.play()
        return captures
    
    def _recalculate_utilities(self, new_partiton):
        # we only ever add a defense policy to the policy matrix, so we can just create a new row and add it
        new_row = np.zeros((1, len(self.attacker_policies)), dtype=np.float32)
        for policy_id, policy_name in self.attacker_policies.items():
            new_row[0,policy_id] = np.mean([self._simulate(new_partiton, policy_name) for _ in range(self.utility_expectation_horizon)])
        self.utility_matrix = np.concat((self.utility_matrix, new_row), axis=0)

    def _form_subgame_matrix(self): # theres probably some clever way to do this 
        subgame_payoffs = np.zeros((len(self.subgame_polices['d']), len(self.subgame_polices['a'])))

        for p_policy_index in range(subgame_payoffs.shape[0]):
            for a_policy_index in range(subgame_payoffs.shape[1]):
                subgame_payoffs[p_policy_index, a_policy_index] = self.utility_matrix[self.subgame_polices['d'][p_policy_index], self.subgame_polices['a'][a_policy_index]]
        return subgame_payoffs
    
    # learning loop
    def _create_defense_policy(self, training_args : dict, subgame_policy_dict : dict, iter_number: int) -> list[list[int]]:
        num_envs = training_args.get('num_envs', 8)
        masked_env = make_vec_env(
            env_id=PartitionEnv,
            env_kwargs={
                'league' : SimulatorEnv(self.graph_config_name, self.defender_policy_name),
                'mixed_policy' : self.attacker_vector,
                'policy_map' : subgame_policy_dict
            },
            n_envs=num_envs,
            vec_env_cls=SubprocVecEnv
        )
        graph_kwargs = {
            'num_nodes' : self.num_nodes,
            'num_defenders' : self.num_defenders,
            # 'num_attacker_policies' : len(subgame_policy_dict),
            'num_attacker_policies' : len(self.attacker_policies),
            'feature_dim' : 8,
            'pos_dim' : 13,
            'hidden_dim' : training_args.get('hidden_dim', 64),
            'edge_index' : self.edge_index,
            'lap_pos' : self.lap_pos,
            'walk_pos' : self.walk_pos
        }
        policy_kwargs = dict(graph_kwargs=graph_kwargs)

        def linear_schedule(init_value):
            def func(progress_remaining):
                return progress_remaining * init_value
            return func


        model = MaskablePPO(
            policy=MaskableGNNPolicy,
            env=masked_env,
            policy_kwargs=policy_kwargs,
            n_steps=training_args.get('n_steps',250),
            batch_size=training_args.get('batch_size',256),
            learning_rate=linear_schedule(training_args.get('learning_rate',3e-4)),
            gamma = 1.0,
            gae_lambda=1.0,
            normalize_advantage=True,
            target_kl= training_args.get('target_kl',0.015),
            clip_range=training_args.get('clip_range',0.15),
            ent_coef=training_args.get('ent_coef',0.01),
            verbose=training_args.get('verbose',0),
            tensorboard_log=training_args.get('tensorboard_log','ppo_tensorboard_logs'),
        )

        ### callbacks
        anneal_callback = RewardAnnealingCallback()
        checkpoint_callback = CheckpointCallback(save_path=training_args.get('save_path','checkpoints'), iter_num=iter_number, verbose=0)
        early_stop_callback = EarlyStoppingCallback(-0.15)
        callbacks = [anneal_callback, checkpoint_callback, early_stop_callback]
        # callbacks = [anneal_callback, checkpoint_callback]
        if training_args.get('tensorboard_log','a') is not None: # unless tblog is explicitly set to None
            callbacks.append(LoggingCallback())
        callbacks = CallbackList(callbacks)

        ### training
        timesteps = training_args.get('timesteps', 2_500_000)
        model.learn(total_timesteps=timesteps, callback=callbacks)

        return MaskablePPO.load(training_args.get('save_path','checkpoints') + f'/best_model-iter-{iter_number}') # return the best model

    def _get_graph_info(self, lap_dim=8, walk_dim=5):
        G : nx.Graph = self.simulator.undirected_view
        top_row, bottom_row = [], []
        for edge in G.edges:
            u, v, _ = edge
            top_row += [u, v]
            bottom_row += [v, u]
        edge_index = torch.tensor([top_row, bottom_row], dtype=torch.long)
        dummy = torch.ones((G.number_of_nodes(), 1))
        data = Data(dummy, edge_index)
        transform = Compose([AddLaplacianEigenvectorPE(k=lap_dim, attr_name='lap'), AddRandomWalkPE(walk_length=walk_dim, attr_name='walk')])
        data = transform(data)
        return edge_index, data.lap, data.walk

    # nash calculation with cvxpy
    # returns game value
    def _calculate_subgame_nash(self, subgame_matrix : np.ndarray)-> float:

        # helper function to round off policies
        def threshold_and_normalize(raw_vector, tolerance=1e-5):
            clean_vector = np.where(raw_vector < tolerance, 0.0, raw_vector)
            clean_vector= clean_vector / np.sum(clean_vector)
            return np.clip(clean_vector, 0, 1)

        # calculate policy for attacker, maximizer
        z = cp.Variable((subgame_matrix.shape[1]))
        v_max = cp.Variable(1)
        maximizer_objective = cp.Maximize(v_max)
        maximizer_constraints = [
            z >= 0,
            cp.sum(z) == 1,
            cp.matmul(subgame_matrix, z) >= v_max
        ]
        maximizer_problem = cp.Problem(maximizer_objective, maximizer_constraints)
        maximizer_problem.solve()

        # minimizer policy 
        y = cp.Variable((subgame_matrix.shape[0]))
        v_min = cp.Variable(1)
        minimizer_objective = cp.Minimize(v_min)
        minimizer_constraints = [
            y >= 0,
            cp.sum(y) == 1,
            cp.matmul(subgame_matrix.T, y) <= v_min
        ]
        minimizer_problem = cp.Problem(minimizer_objective, minimizer_constraints)
        minimizer_problem.solve()
        self.partition_vector = threshold_and_normalize(y.value)
        self.attacker_vector = threshold_and_normalize(z.value)
        return float(v_min.value[0])

    
    # for each attacker policy, get its utility (from the util matrix) against any defense policy that has a non-zero value in the subgame defense policy.
    # weight the mean value of these sims by the defense policy weight 
    def _find_attacker_br(self):
        full_defense_policy = self._to_full_policy(self.partition_vector, player='d')
        policy_values = full_defense_policy @ self.utility_matrix  # should be a len(attacker policy vector)
        best_policy = np.argmax(policy_values)
        return int(best_policy)
    
    # call to the learning loop and rollout
    def _find_defense_br(self, training_args : dict, iter_number: int):
        subgame_policy_dict = {policy_id : self.attacker_policies[policy_id] for policy_id in self.subgame_polices['a']}
        if self.logging:
            logging.info(f'Defense BR considers policies: {list(subgame_policy_dict.values())}')
        model = self._create_defense_policy(training_args, subgame_policy_dict, iter_number)
        ### use model to produce partition
        env = PartitionEnv(
            SimulatorEnv(self.graph_config_name, self.defender_policy_name),
            mixed_policy=self.attacker_vector,
            policy_map=subgame_policy_dict
        )
        obs, _ =  env.reset()
        done = False
        partition = {d : set(range(self.num_nodes)) for d in range(self.num_defenders)}
        while not done:
            valid_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks = valid_masks, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            if action != env.commit_action:
                defender_id = action // env.num_nodes
                node_id = action % env.num_nodes
                partition[defender_id].discard(node_id)
            
            obs, _, done, _, _ = env.step(action)
        return [list(partition[k]) for k in partition.keys()]

    def _log_iter_to_json(self, iteration: int, current_utility : float, loop_time: float, run_dir: str):
        log_data = {
            'iteration' : iteration,
            'current_expected_utility' : current_utility,
            'loop_wall_clock_seconds' : loop_time,

            'subgame_attackers' : self.subgame_polices['a'],
            'subgame_defenders' : self.subgame_polices['d'],

            'attacker_nash_vector' : self.attacker_vector.tolist(),
            'defender_nash_vector' : self.partition_vector.tolist(),
            'full_utility_matrix' : self.utility_matrix.tolist(),

            'partition_policies' : {str(k) : v for k,v in self.partition_policies.items()},
            'attacker_policies' : {str(k) : v for k,v in self.attacker_policies.items()},
        }
        os.makedirs(run_dir, exist_ok=True)
        file_path = os.path.join(run_dir, f'psro_meta_iter_{iteration:03d}.json')
        with open(file_path, 'w') as f:
            json.dump(log_data, f, indent=4)
    
if __name__ == '__main__':
    training_args = {
        'timesteps' : 30_000,
        'clip_range' : 0.2,
        'target_kl' : 0.05,
        'ent_coef' : 0.05,
        'num_envs' : 1,
        'n_steps' : 500

    }

    test = PSRO(
        graph_config_name='small-grid.yml',
        defender_policy_name='Example_Def',
        utility_expectation_horizon=10
    )
    test.solve(training_args=training_args)