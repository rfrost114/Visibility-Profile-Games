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
    # print(f'D: {action}')
    state['action'] = action



def policy(
        agent_name,
        current_node,
        flag_positions,
        attacker_positions,
        defender_positions,
        graph : nx.Graph
):

    
    closest_attacker = None
    min_distance = float('inf')
    
    # Find the closest attacker based on shortest path distance
    for attacker in attacker_positions:
        for flag in flag_positions:
            try:
                # Compute the unweighted shortest path length to the attacker
                dist = nx.shortest_path_length(
                    graph, source=attacker, target=flag
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_attacker = attacker
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Skip if no path exists or node is not found
                continue

    if closest_attacker is None:
        # Fallback: move to a random neighboring node if no attacker is found
        neighbor_data = list(graph.neighbors(current_node)) + [current_node]
        action = random.choice(neighbor_data)
        action = current_node
    
    else:
        try:
            # Determine the next node towards the closest attacker
            path = nx.shortest_path(graph, source=current_node, target=closest_attacker)
            # print(f'{path=}')
            action = path[0] if len(path)==1 else path[1]
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            # Handle cases where the path cannot be found
            # print(f"No path found from blue agent at node {current_node} to attacker at node {closest_attacker}: {e}")
            neighbor_data = list(graph.neighbors(current_node)) + [current_node]
            action = random.choice(neighbor_data)
            action = current_node
    
    return action

def map_strategy(agent_config):
    """
    Maps each defender agent to the defined strategy.
    
    Parameters:
        agent_config (dict): Configuration dictionary for all agents.
        
    Returns:
        dict: A dictionary mapping agent names to their strategies.
    """
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
