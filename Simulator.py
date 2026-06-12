import importlib 
import os
import pathlib
import traceback
import networkx as nx
import gamms
import numpy as np
from typing import Optional, Tuple




from lib.core.core import *
from lib.utils.file_utils import get_directories, export_graph_config
from lib.utils.config_utils import load_configuration, create_context_with_sensors
from lib.utils.sensor_utils import create_static_sensors
from lib.utils.game_utils import *
    # import lib.core.time_logger as TLOG
import matplotlib.pyplot as plt



class SimulatorEnv:

    def __init__(
            self,
            config_name : str,
            defender_policy_name : str,
            visualization: bool = False,
            debug : bool = False
        ):
        self.config_name = config_name
        self.defender_policy_name = defender_policy_name
        self.visualization = visualization
        self.debug = debug

        # static gamms infor
        self.root_path = str(pathlib.Path(__file__).resolve().parent)
        dirs = get_directories(self.root_path)
        self.config = load_configuration(self.config_name, dirs, self.debug)

        self.max_time = self.config.get("game", {}).get("max_time", 1000)
        self.flag_positions = self.config.get("game", {}).get("flag", {}).get("positions", [])
        self.flag_weights = self.config.get("game", {}).get("flag", {}).get("weights")
        self.interaction_config = self.config.get("game", {}).get("interaction", {})
        self.payoff_config = self.config.get("game", {}).get("payoff", {})


        self.G = export_graph_config(self.config, dirs, debug)
        self.undirected_view = nx.to_undirected(self.G)
        self.diameter = nx.diameter(self.undirected_view)
        self.num_nodes = self.G.number_of_nodes()
        self.save_partitions = False # save graph images
        # print(list(self.G.nodes(data=True)))
        # self.pos = {}
        # for idx, node_data in list(self.G.nodes(data=True)):
        #     print(node_data['x'])
        #     self.pos[idx] = np.array([
        #         node_data['x'],
        #         node_data['y']
        #     ])
        # # non static behavors
        self.reset()


    def reset(self):
        self.partition_assignment = None
        self.time_counter = 0
        self.tags = 0
        self.captures = 0
        self.payoff = 0
        self.ctx = None
        self.partition_assignment = {}
        self.paralyzed_defender_penalty = 0
        self.final_distance = float('inf')
        self.most_recent_locations = {'a' : [], 'd' : []}

        static_sensors = create_static_sensors()

        self.ctx = create_context_with_sensors(self.config, self.G, visualization=self.visualization, static_sensors=static_sensors, debug=self.debug)
        self.agent_config, self.agent_params_dict = initialize_agents(self.ctx, self.config)
        self.d_locs = [self.agent_config[name]['start_node_id'] for name in self.agent_config.keys() if name.startswith('d')]
        self.a_locs = [self.agent_config[name]['start_node_id'] for name in self.agent_config.keys() if name.startswith('a')]
        self.num_defenders = len(self.d_locs)
        if self.save_partitions:
            
            self.pos = {}
            for idx, node_data in list(self.G.nodes(data=True)):
                self.pos[idx] = np.array([
                    node_data['x'],
                    node_data['y']
                ])

    

    def set_attacker_heuristic(self, attacker_policy_name : str):

        attacker_policy = importlib.import_module(f'policies.attacker.{attacker_policy_name}')
        defender_policy = importlib.import_module(f'policies.defender.{self.defender_policy_name}')


        # if self.debug:
        #     atk_name = attacker_policy.__name__.split(".")[-1] 
        #     def_name = defender_policy.__name__.split(".")[-1]
        #     print(f'{attacker_policy_name=} {self.defender_policy_name=}')
        #     print(f'{atk_name=} {def_name=}')
        assign_strategies(self.ctx, self.agent_config, attacker_policy, defender_policy)
        configure_visualization(self.ctx, self.agent_config, self.config)
        initialize_flags(self.ctx, self.config)

        # at this point all of the initalization is done just need to set the partition 
    
    def set_partition(self, partitions : list[list[int]]):
        # we will create a dictionary with keys for each defender agent name where the value is the nodes visible to that agent
        defender_names = [agent.name for agent in self.ctx.agent.create_iter() if str(agent.name).startswith('d')]
        max_captures = len(self.a_locs)
        para_penalty = max_captures + 1

        # per_defender_penalty = max_para_penalty / len(defender_names) # pentaly incured by each paralyzed defender 


        for index, defender in enumerate(defender_names):
            # graph = nx.to_undirected(self.G)
            
            node_list = set(partitions[index])
            node_list.add(self.d_locs[index]) # do stop a crash we always ensure that the starting position of the defender is within its partition
            # print(node_list)
            partitioned_graph = nx.Graph(nx.induced_subgraph(self.undirected_view, node_list))
            if self.save_partitions:
                nx.draw_networkx(partitioned_graph, self.pos, node_size=10)
                plt.savefig(f'p{index+1}g.png')
            self.partition_assignment[defender] = {
                'nodelist' :  node_list,
                'graph' : partitioned_graph,

            }
        
            if not len(list(partitioned_graph.neighbors(self.d_locs[index]))): # if the start node has no neighbours impose the penalty
                self.paralyzed_defender_penalty += para_penalty


        if self.debug:
            print(self.partition_assignment)
    
    def play(self):
        # Check initial interactions before any moves
        
        init_caps, init_tags, a_count, d_count, _, _ = check_agent_interaction(self.ctx, self.G, self.agent_params_dict, self.flag_positions, self.interaction_config, self.time_counter)
        self.captures += init_caps
        self.tags += init_tags
        if check_termination(self.time_counter, self.max_time, a_count, d_count):
            self._min_dist_at_termination()
            return self.captures

        while not self.ctx.is_terminated():
            self.time_counter += 1
            next_actions = {}

            try:
                for agent in self.ctx.agent.create_iter():
                    state = agent.get_state()
                    state.update({
                        "flag_pos": self.flag_positions,
                        "flag_weight": self.flag_weights,
                        "agent_params": self.agent_params_dict.get(agent.name, {}),
                        "time": self.time_counter,
                        "payoff": self.payoff,
                        "name": agent.name,
                        "agent_params_dict": self.agent_params_dict,
                        "partition_data" : self.partition_assignment.get(agent.name, {}) # we add the partition data if it exists for the agent, only when agent is defender
                        })
                    

                    if hasattr(agent, "strategy") and agent.strategy is not None:
                        try:
                            agent.strategy(state)
                            # partition_list = self.partition_assignment.get(agent.name)
                            # if partition_list is not None: # we are looking at a defender agent
                            #     agent.strategy(state, partition_list)
                            # else: # we are looking at an attacker
                            #     agent.strategy(state)
                            
                        except Exception as e:
                            error(f"Error executing strategy for {agent.name}: {e}")
                            traceback.print_exc()
                    else:
                        node = self.ctx.visual.human_input(agent.name, state)
                        state["action"] = node
                    next_actions[agent.name] = state["action"]
                    
            except Exception as e:
                error(f"An error occurred during agent turn: {e}")
                raise e

            # Update agents with their actions
            for agent in self.ctx.agent.create_iter():
                
                state = agent.get_state()
                state["action"] = next_actions.get(agent.name, state.get("action", None))
                agent.set_state()

            self.ctx.visual.simulate()
            # log locations
            a, d = set(), set()
            for agent in self.ctx.agent.create_iter():
                state = agent.get_state()
                if str(agent.name).startswith('a'):
                    a.add(state['curr_pos'])
                else:
                    d.add(state['curr_pos'])
            self.most_recent_locations['a'] = a
            self.most_recent_locations['d'] = d

            captures, tags, a_count, d_count, _, _ = check_agent_interaction(self.ctx, self.G, self.agent_params_dict, self.flag_positions, self.interaction_config, self.time_counter)
            self.captures += captures
            self.tags += tags

            if check_termination(self.time_counter, self.max_time, a_count, d_count):
                self._min_dist_at_termination()
                    
                break
        return self.captures

    def get_score(self):
        return self.captures
    
    def _min_dist_at_termination(self):
        if not len(self.most_recent_locations['a']):
            self.final_distance = 0
        else:
            for attacker_position in self.most_recent_locations['a']:
                for defender_position in self.most_recent_locations['d']:
                    self.final_distance = min(
                        self.final_distance,
                        nx.shortest_path_length(self.undirected_view, attacker_position, defender_position)
                    )
                




if __name__ == '__main__':
    # testsim = SimulatorEnv('example_2v2.yml', 'Example_Def', visualization=True)
    policy_names = ['Example_Def', 'Patrolling_Def']
   
    testsim = SimulatorEnv('100-node.yml', policy_names[1], visualization=True)
    # print(testsim.G.nodes)
    testsim.set_attacker_heuristic('Ret_Atk')
    partition = [
        # list(range(12))
        # [0,1,2,3,5,6,7,8,9,11]
        # [0,2,3,4,5,6,7,8,9,10,11]
        #[1,2,6,8,9,10,11]
        # [0, 2, 3, 4, 5, 6, 7, 9, 10, 18],
        # [5, 8, 12, 19, 22, 28, 77, 78]
        
        
        # [i for i in range(100) if i not in [9, 10, 6, 5]],

        # list(range(19)) + list(range(20,77))+ list(range(78,81))+ list(range(82,100)),
        
    ]
    partition = [[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 79, 80, 82, 83, 85, 87, 88, 89, 90, 91, 92, 94, 95, 97, 98, 99], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99]]
    testsim.set_partition(partition)
    testsim.play()
    print(testsim.get_score())
    # print(testsim.paralyzed_defender_penalty)
    # print(testsim.final_distance)
    # print(testsim.most_recent_locations)
    # testsim.reset()

    # partition = [
    #     [1],
    #     list(range(50,100)),
    # ]

    # testsim.set_attacker_heuristic('Example_Atk')
    # testsim.set_partition(partition)
    # testsim.play()
    # print(testsim.get_score())