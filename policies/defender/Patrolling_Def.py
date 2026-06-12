import random
import networkx as nx
from lib.utils.sensor_utils import extract_sensor_data, extract_neighbor_sensor_data

def strategy(state : dict):

    DEBUG = False
    # print(state.keys())
    agent_name = state['name']
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    agent_graph : nx.Graph = state['partition_data']['graph']
    partition_list : set = state['partition_data']['nodelist']
    # Extract positions of attackers and defenders from sensor data
    attacker_positions, defender_positions = extract_sensor_data(
        state, flag_positions, flag_weights, agent_params
    )


    # # get the partitioned graph or create it if it does not exist
    # agent_graph = state.get(f'{agent_name}_partition')
    # # print(f'{state["time"]= } {agent_graph=}')
    # if agent_graph is None:
    #     # print('this is being done!!!!!!')
    #     graph = nx.to_undirected(agent_params.map.graph)
    #     agent_graph = nx.Graph(nx.induced_subgraph(graph, partition_list))
    #     state[f'{agent_name}_partition'] = agent_graph
    # # print(state.keys())
    
    filtered_attackers = list(set(attacker_positions).intersection(partition_list))
    filtered_defenders = list(set(defender_positions).intersection(partition_list))
    filtered_flags = list(set(flag_positions).intersection(partition_list))
    # print(f'{flag_positions=}')
    # print(f'{partition_list=}')
    if DEBUG:
        print(f'I am Agent {agent_name}')
        print(f'The graph I see has {agent_graph.number_of_nodes()} nodes')
        print(f'I see flags at {filtered_flags} and attackers at {filtered_attackers}')
        print(f'My partition is {partition_list}')
    # print(agent_graph.nodes)
    action = policy(
        agent_name=agent_name,
        current_node=current_node,
        flag_positions=filtered_flags,
        attacker_positions=filtered_attackers,
        defender_positions=filtered_defenders,
        graph=agent_graph
    )
    # print(f'{action=}')
    state['action'] = action


def policy(
        agent_name,
        current_node,
        flag_positions,
        attacker_positions,
        defender_positions,
        graph : nx.Graph   
):
    key_nodes = find_key_nodes(graph, flag_positions)
    patrol_path = get_patrol_path(graph, flag_positions, key_nodes)
    
    for attacker in attacker_positions:
        for flag in flag_positions:
            # print(f'{attacker} {flag} {attacker_positions} {flag_positions}')
            if nx.shortest_path_length(graph, source=attacker, target=flag) <= len(attacker_positions + flag_positions): 
                closest_defender = min(
                    defender_positions,
                    key=lambda d: nx.shortest_path_length(graph, source=d, target=attacker),
                    default=None
                )
                if closest_defender == current_node:
                    try:
                        # path = nx.shortest_path(graph, source=current_node, target=attacker)

                        # next_node = path[0] if len(path)==1 else path[1]
                        # # print('inital')
                        # action = next_node
                        next_node = shortest_path_to(graph, current_node, attacker, 1)
                        action = next_node
                        return action
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass 
    
    if current_node in patrol_path:
        # print('if')
        next_index = (patrol_path.index(current_node) + 1) % len(patrol_path)
        action = patrol_path[next_index]
    else:
        # print('else')
        nearest_patrol_node = min(patrol_path, key=lambda n: nx.shortest_path_length(graph, source=current_node, target=n))
        path = nx.shortest_path(graph, source=current_node, target=nearest_patrol_node)
        action = path[0] if len(path)==1 else path[1]
    
    return action


def find_key_nodes(graph, flag_positions):
    key_nodes = []
    for flag in flag_positions:
        neighbors = list(graph.neighbors(flag))
        high_degree_neighbors = sorted(neighbors, key=lambda n: graph.degree[n], reverse=True)
        if high_degree_neighbors:
            key_nodes.append(high_degree_neighbors[0])
    return key_nodes

def get_patrol_path(graph, flag_positions, key_nodes):
    if graph.is_directed():
        graph = graph.to_undirected()  
    
    subgraph = graph.subgraph(flag_positions + key_nodes)
    mst = nx.minimum_spanning_tree(subgraph)
    # print(f'{flag_positions=}')
    path = list(nx.dfs_preorder_nodes(mst, source=max(flag_positions)))
    return path + [path[0]]  # Close the loop

def shortest_path_to(graph, source_node, target_node, speed=1 ):
    if not isinstance(target_node, (list, tuple, set)):
        target_nodes = [target_node]
    else:
        target_nodes = list(target_node)
    best_path = None
    best_length = float("inf")

    for t in target_nodes:
        try:
            length, path = nx.single_source_dijkstra(graph, source=source_node, target=t, weight="length")
            if length < best_length:
                best_length = length
                best_path = path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print('no path found in graph')
            continue
    
    if best_path is None:
        return None
    index = speed if speed < len(best_path) else len(best_path) - 1
    return best_path[index]

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
