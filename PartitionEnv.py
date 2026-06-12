import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Simulator import SimulatorEnv
import networkx as nx

class PartitionEnv(gym.Env):

    def __init__(
            self,
            league : SimulatorEnv,
            mixed_policy : np.ndarray,
            policy_map : dict[int : str]
    ):
        super().__init__()
        self.leagueSim = league
        # we call this here just so num defenders populates, will be called again later
        self.leagueSim.reset()
        self.num_defenders = self.leagueSim.num_defenders
        self.num_nodes = self.leagueSim.num_nodes

        self.mixed_policy = mixed_policy
        self.policy_map = policy_map # dict of indicies and file policy names 
        self.base_featues = None
        self.beta = 1

        #0-(n-1) defnder 1, n-(2n-1) def 2, ...
        # n*k -> commit
        self.action_space = spaces.Discrete((self.num_nodes * self.num_defenders) + 1)
        self.commit_action = self.num_nodes * self.num_defenders


        # observation is 
        #1,0,0,0 : d, 0100: a, 0010: f, 0001: n
        # self.observation_space = spaces.Box(
        #     low=0.0,
        #     high=1.0,
        #     shape=(self.num_nodes, 8 +self.num_defenders),
        #     dtype=np.float32
        # )
        self.observation_space = spaces.Dict({
            'public_obs' : spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_nodes, 8 + self.num_defenders),
                dtype=np.float32
            ),
            'private_obs' : spaces.Discrete(len(self.policy_map)) # we let the centralized critic know what policy is being played 
        })

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        # print('reset called')
        # reset the simulator
        self.leagueSim.reset()
        # sample an attacker policy from the current set of mixed policies 
        policy_choice = np.random.choice(list(self.policy_map.keys()), p=self.mixed_policy)
        self.policy_choice_id = policy_choice
        self.policy_name = self.policy_map[policy_choice]
        # initialize the chosen heuristic 
        self.leagueSim.set_attacker_heuristic(self.policy_map[policy_choice])
        # print(f'Chose Policy: {self.policy_map[policy_choice]}')
        base_features = self._build_base_features()
        active_nodes = np.ones((self.num_nodes, self.num_defenders), dtype=np.float32)
        self.current_obs = np.concatenate((base_features, active_nodes), axis=1)

        obs_dict = {
            'public_obs' : self.current_obs.copy(),
            'private_obs' : self.policy_choice_id
        }

        return obs_dict, {}

    
    def _build_base_features(self):
        
        # if self.base_featues is None:
        observation = np.zeros((self.leagueSim.num_nodes, 4), dtype=np.float32)

        for d_loc in self.leagueSim.d_locs : observation[d_loc, 0] = 1.0
        for a_loc in self.leagueSim.a_locs : observation[a_loc, 1] = 1.0
        for f_loc in self.leagueSim.flag_positions : observation[f_loc, 2] = 1.0

        pois = self.leagueSim.d_locs + self.leagueSim.a_locs + self.leagueSim.flag_positions

        degree_features = []
        for node in range(self.num_nodes):
            if node not in pois:
                observation[node, 3] = 1.0
            
            degree = self.leagueSim.undirected_view.degree(node)
            neighbour_degrees = [self.leagueSim.undirected_view.degree(i) for i in self.leagueSim.undirected_view.neighbors(node)]
            features = [degree, np.mean(neighbour_degrees), min(neighbour_degrees), max(neighbour_degrees)]
            degree_features.append(features)
        
        observation = np.concatenate((observation, np.array(degree_features).astype(np.float32)), axis=1)
        # self.base_featues = observation
        
        return observation
    
    def step(self, action): # action is now an int between 1 and (num_nodes * num_defs) + 1



        if action == self.commit_action:
            partitions = self.current_obs[:,8:] # just the last four cols of the observation
            node_lists = [list(map(int, (np.where(partitions[:,i]==1)[0]))) for i in range(self.num_defenders)]
            self.leagueSim.set_partition(node_lists)
            captures = self.leagueSim.play()


            terminal_reward = -1 * captures
            # print(f'{self.beta=}')

            dense_reward = -1 * (self.leagueSim.final_distance / self.leagueSim.diameter)
            scaled_dense = (self.beta * dense_reward)
            # print(f'{scaled_dense=} {beta=} {dense_reward=}')
            final_reward = terminal_reward + scaled_dense

            info = {
                'captures' : captures,
                'terminal_reward' : terminal_reward,
                'final_reward' : final_reward,
                'dense_rewards' : dense_reward,
                'beta' : self.beta
            }

            obs_dict = {
                'public_obs' : self.current_obs.copy(),
                'private_obs' : self.policy_choice_id
            }

            return obs_dict, final_reward, True, False, info

        else:
            defender_id = action // self.num_nodes
            node_id = action % self.num_nodes

            self.current_obs[node_id, 8 + defender_id] = 0.0

            step_penalty = 0

            obs_dict = {
                'public_obs' : self.current_obs.copy(),
                'private_obs' : self.policy_choice_id
            }

            return obs_dict, step_penalty, False, False, {}

        
    
    def action_masks(self): # boolean mask for valid actions
        mask = np.ones(self.action_space.n, dtype=bool)
        # total_dropped = np.sum(self.current_obs[:, 4:]==0)


        for defender_id in range(self.num_defenders):
            articulation_points = self._find_articulation_points(self.current_obs[:, defender_id + 8])
            # find the included attacker nodes for the defender
            remaining_attackers, remaining_flags = self._find_remaining_pois(self.current_obs[:, defender_id + 8])

            for node_id in range(self.num_nodes):
                action_idx = (defender_id * self.num_nodes) + node_id

                # do not drop nodes that are already dropped
                if self.current_obs[node_id, defender_id + 8] == 0.0:
                    mask[action_idx] = False
                
                # do not drop your starting location
                if node_id == self.leagueSim.d_locs[defender_id]:
                    mask[action_idx] = False

                # do not drop nodes that would disonnect the partition...
                if node_id in articulation_points:
                    mask[action_idx] = False
                
                if len(remaining_attackers) == 1 and (node_id in remaining_attackers):
                    mask[action_idx] = False
                
                if len(remaining_flags) == 1 and (node_id in remaining_flags):
                    mask[action_idx] = False

        # # commit action should alway be valid
        mask[self.commit_action] = True
        # if total_dropped < 2:
        #     mask[self.commit_action] = False
        # else:
        #     mask[self.commit_action] = True
        
        return mask

    def set_beta(self, new_beta : float):
        self.beta = new_beta

    def _find_articulation_points(self, partition_vector : np.ndarray): 
        current_nodes = [i for i in range(self.num_nodes) if (partition_vector[i]==1)]
        subgraph = self.leagueSim.undirected_view.subgraph(current_nodes)
        return list(nx.articulation_points(subgraph))
    
    def _find_remaining_pois(self, partition_vector : np.ndarray):
        current_nodes = set(i for i in range(self.num_nodes) if (partition_vector[i]==1))
        seen_attackers = current_nodes.intersection(set(self.leagueSim.a_locs))
        seen_flags = current_nodes.intersection(set(self.leagueSim.flag_positions))
        return seen_attackers, seen_flags
    

if __name__ == '__main__':
    Sim = SimulatorEnv(
        config_name='100-node.yml',
        defender_policy_name='Patrolling_Def'
    )

    mixed_policy = np.array([0.33,0.34,0.33])
    attacker_policy_dict = {
        0 : 'Example_Atk',
        1 : 'MPC_Atk',
        2 : 'Ret_Atk'
    }

    test_environment = PartitionEnv(
        league=Sim,
        mixed_policy=mixed_policy,
        policy_map=attacker_policy_dict
    )
    
    rewards = []
    for _ in range(1000):
        obs, info = test_environment.reset()
        # action = np.ones((test_environment.num_nodes * test_environment.num_defenders)+1)
        action = test_environment.commit_action
        obs, reward, _, _, _ = test_environment.step(action)
        policy_name = test_environment.policy_name
        # print(obs)
        rewards.append((policy_name, reward))
        # print(reward)
    mean_reward = np.mean([r[1] for r in rewards])
    for j, (name, reward) in enumerate(rewards[:50]):
        print(f'Iter {j} - Policy: {name} Reward: {reward}')
    print(f'Mean Reward: {mean_reward}')

    # print(rewards)