# attacker_strategy
import networkx as nx
import pickle
from lib.utils.sensor_utils import extract_sensor_data, extract_neighbor_sensor_data
import os

# Load distance lookup table
#with open("distance_lookup.pkl", "rb") as f:
#    distance_lookup = pickle.load(f)

_distance_lookup = None

def get_distance_lookup(graph):
    global _distance_lookup
    if _distance_lookup is None:
        if os.path.exists("distance_lookup.pkl"):
            with open("distance_lookup.pkl", "rb") as f:
                _distance_lookup = pickle.load(f)
        else:
            #print("[Info] Generating distance lookup table...")
            _distance_lookup = dict(nx.all_pairs_shortest_path_length(graph))
            _distance_lookup = {src: dict(dst_lengths) for src, dst_lengths in _distance_lookup.items()}
            with open("distance_lookup.pkl", "wb") as f:
                pickle.dump(_distance_lookup, f)
    return _distance_lookup


def attacker_stage_cost(pos, defenders, flags, lookup, w_goal=3.0, w_risk=2.0, epsilon=1e-2):
    dist_to_flag = min(lookup[pos][f] for f in flags)
    risk_penalty = sum(1 / (lookup[pos][d] + epsilon) for d in defenders)
    return w_goal * dist_to_flag + w_risk * risk_penalty

def attacker_terminal_cost(pos, defenders, flags, lookup, lambda_weight=5.0, epsilon=1e-2):
    #if pos in flags:
    #    return -10
    min_flag_dist = min(lookup[pos][f] for f in flags)
    risk_term = sum(1 / (lookup[pos][d] + epsilon) for d in defenders)
    return min_flag_dist + lambda_weight * risk_term

def evaluate_attacker_lookahead(pos, defenders, flags, graph, lookup, depth, beta=1.0):
    if depth == 0:
        return attacker_terminal_cost(pos, defenders, flags, lookup)
    
    best_value = float('inf')
    for neighbor in graph.neighbors(pos):
        stage = attacker_stage_cost(neighbor, defenders, flags, lookup)
        future = evaluate_attacker_lookahead(neighbor, defenders, flags, graph, lookup, depth - 1, beta)
        total = stage + beta * future
        best_value = min(best_value, total)
    
    return best_value

def strategy(state):
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    agent_params = state['agent_params']
    depth = 5
    attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, state['flag_weight'], agent_params)
    graph = agent_params.map.graph
    distance_lookup = get_distance_lookup(graph)

    best_move = current_node
    best_score = float('inf')

    for neighbor in graph.neighbors(current_node):
        total_cost = evaluate_attacker_lookahead(neighbor, defender_positions, flag_positions, graph, distance_lookup, depth)
        if total_cost < best_score:
            best_score = total_cost
            best_move = neighbor

    state['action'] = best_move

def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}

