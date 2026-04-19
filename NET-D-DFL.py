import time
import functools
import torch
import random
import copy
import math
import statistics
import os
import pandas
import torchvision
import torchvision.transforms as data_transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as neural_network
import torch.optim as optimization_module
import networkx as graph_library
import matplotlib.pyplot as plotting_module
import warnings

warnings.filterwarnings("ignore")

def count_epidemic_states(network_graph, total_nodes):
    susceptible_count = 0
    active_count = 0
    recovered_count = 0
    for node_index in network_graph:
        if network_graph.nodes[node_index]["state"] == "I":
            susceptible_count += 1
        if network_graph.nodes[node_index]["state"] == "A":
            active_count += 1
        if network_graph.nodes[node_index]["state"] == "R":
            recovered_count += 1
    return susceptible_count, total_nodes - susceptible_count - recovered_count, recovered_count

class DigitClassifier(neural_network.Module):
    def __init__(self):
        super().__init__()
        self.flatten_layer = neural_network.Flatten()
        self.fully_connected_one = neural_network.Linear(28 * 28, 200)
        self.activation_one = neural_network.ReLU()
        self.fully_connected_two = neural_network.Linear(200, 200)
        self.activation_two = neural_network.ReLU()
        self.output_layer = neural_network.Linear(200, 10)

    def forward(self, input_tensor):
        flattened_tensor = self.flatten_layer(input_tensor)
        hidden_one = self.activation_one(self.fully_connected_one(flattened_tensor))
        hidden_two = self.activation_two(self.fully_connected_two(hidden_one))
        return self.output_layer(hidden_two)

def evaluate_node_accuracy(classification_model, testing_data_loader):
    total_correct = 0
    total_samples = 0
    classification_model.eval()
    with torch.no_grad():
        for features, labels in testing_data_loader:
            predictions = classification_model(features)
            predicted_classes = torch.argmax(predictions, dim=1)
            total_correct += torch.sum(predicted_classes == labels).item()
            total_samples += labels.size(0)
    classification_model.train()
    return total_correct / total_samples

def update_epidemic_states(network_graph, accuracy_history, current_round, state_change_flags, saturation_tracker, saturation_time):
    working_graph = network_graph.copy()
    for node_index in working_graph:
        if working_graph.nodes[node_index]["state"] == "A":
            active_recovered_neighbors = 0
            for neighbor_index in working_graph.adj[node_index]:
                if working_graph.nodes[neighbor_index]["state"] != "I":
                    active_recovered_neighbors += 1
            if active_recovered_neighbors == working_graph.degree[node_index]:
                saturation_tracker[node_index] += 1
            else:
                state_change_flags[node_index] = 0
            if saturation_tracker[node_index] == 1:
                saturation_time[node_index] = current_round
            if saturation_time[node_index] <= current_round and current_round > 0:
                random_threshold = random.random()
                accuracy_delta = accuracy_history[node_index][current_round] - accuracy_history[node_index][current_round - 1]
                recovery_probability = 1.0 / ((1.0 + math.exp(accuracy_delta)) * 50.0)
                if random_threshold < recovery_probability:
                    network_graph.nodes[node_index]["state"] = "R"
                    state_change_flags[node_index] = 1
        if working_graph.nodes[node_index]["state"] == "I":
            random_threshold = random.random()
            infection_probabilities = []
            for neighbor_index in working_graph.adj[node_index]:
                accuracy_difference = accuracy_history[neighbor_index][current_round] - accuracy_history[node_index][current_round]
                infection_chance = 1.5 - 1.5 / (1.0 + math.exp(-accuracy_difference))
                if working_graph.nodes[neighbor_index]["state"] == "A" and accuracy_difference > 0:
                    infection_probabilities.append(infection_chance)
            probability_count = len(infection_probabilities)
            if probability_count > 0:
                combined_survival = 1.0
                for probability_value in infection_probabilities:
                    combined_survival *= probability_value
                if random_threshold < (1.0 - combined_survival):
                    network_graph.nodes[node_index]["state"] = "A"
                    state_change_flags[node_index] = 0
        if working_graph.nodes[node_index]["state"] == "R":
            random_threshold = random.random()
            infection_probabilities = []
            for neighbor_index in working_graph.adj[node_index]:
                accuracy_difference = accuracy_history[neighbor_index][current_round] - accuracy_history[node_index][current_round]
                infection_chance = 1.5 - 1.5 / (1.0 + math.exp(-accuracy_difference))
                gradient_difference = accuracy_history[neighbor_index][current_round] - accuracy_history[neighbor_index][current_round - 1] - accuracy_history[node_index][current_round] + accuracy_history[node_index][current_round - 1]
                if working_graph.nodes[neighbor_index]["state"] == "A" and accuracy_difference > 0 and gradient_difference > 0:
                    infection_probabilities.append(infection_chance)
            probability_count = len(infection_probabilities)
            if probability_count > 0:
                combined_survival = 1.0
                for probability_value in infection_probabilities:
                    combined_survival *= probability_value
                if random_threshold < (1.0 - combined_survival):
                    network_graph.nodes[node_index]["state"] = "A"
                    state_change_flags[node_index] = 0

def execute_net_d_dfl_simulation():
    total_nodes = 10
    communication_rounds = 100
    learning_rate = 0.01
    momentum_value = 0.9
    batch_size = 32

    transform_pipeline = data_transforms.Compose([data_transforms.ToTensor(), data_transforms.Normalize((0.5,), (0.5,))])
    full_training_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_pipeline)
    testing_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_pipeline)
    testing_data_loader = DataLoader(testing_dataset, batch_size=1000, shuffle=False)

    # FIX 1: WS graph requires k to be even and k < n.
    # range(1,10,1) produced invalid odd k values (k=1,3,5,7,9) causing
    # malformed graphs with near-zero edges and zero active-edge counts.
    # Using even values only: 2, 4, 6, 8.
    for graph_parameter in range(2, 10, 2):
        execution_durations = []
        for execution_index in range(1, 10, 1):
            partition_size = len(full_training_dataset) // total_nodes
            dataset_partitions = random_split(full_training_dataset, [partition_size] * total_nodes)
            training_data_loaders = [DataLoader(partition, batch_size=batch_size, shuffle=True) for partition in dataset_partitions]

            classification_models = [DigitClassifier() for index in range(total_nodes)]
            model_optimizers = [optimization_module.SGD(model.parameters(), lr=learning_rate, momentum=momentum_value) for model in classification_models]
            loss_function = neural_network.CrossEntropyLoss()

            initial_weights = [parameter.data.clone() for parameter in classification_models[0].parameters()]
            for node_index in range(1, total_nodes):
                for initial_parameter, node_parameter in zip(initial_weights, classification_models[node_index].parameters()):
                    node_parameter.data.copy_(initial_parameter)

            # FIX 2: graph_parameter is now always even so WS graph is valid.
            network_graph = graph_library.watts_strogatz_graph(total_nodes, graph_parameter, 0.3, seed=123)
            node_sequence = sorted(network_graph.nodes())
            adjacency_matrix = graph_library.to_numpy_array(network_graph, nodelist=node_sequence)

            print(f"\n=== k={graph_parameter} | Run {execution_index} | "
                  f"Edges: {network_graph.number_of_edges()} | "
                  f"Degrees: {[d for _, d in network_graph.degree()]} ===")

            directory_path = './results/BA/' + str(graph_parameter) + "/"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            adjacency_dataframe = pandas.DataFrame(adjacency_matrix)
            adjacency_dataframe.to_excel(directory_path + 'H' + str(execution_index) + ".xlsx")

            for node_index in network_graph:
                network_graph.nodes[node_index]["state"] = "I"

            maximum_degree_node = max(network_graph.degree, key=lambda identifier: identifier[1])[0]
            network_graph.nodes[maximum_degree_node]["state"] = "A"

            global_accuracy_history = []
            population_counts_history = []
            node_state_history = []
            node_specific_accuracy = [[] for index in range(total_nodes)]
            state_change_flags = torch.zeros(total_nodes)
            state_change_history = []
            saturation_tracker = torch.zeros(total_nodes)
            saturation_time = [communication_rounds] * total_nodes

            # ── Communication cost tracking ────────────────────────────────
            # active_edges counts the number of directed model transmissions
            # per round. Each undirected A-A edge = 2 transmissions (both
            # nodes send their model to each other), so we multiply by 2.
            active_edges_per_round = []
            nodes_active_per_round = []
            round_time_history     = []

            start_time = time.time()

            for current_round in range(communication_rounds):
                population_counts = count_epidemic_states(network_graph, total_nodes)
                population_counts_history.append(list(population_counts))

                current_states = []
                for node_index in range(total_nodes):
                    current_states.append(network_graph.nodes[node_index]["state"])
                node_state_history.append(current_states)

                if population_counts[1] == 0:
                    break

                state_change_history.append(state_change_flags.clone())

                for node_index in range(total_nodes):
                    if network_graph.nodes[node_index]["state"] == "A":
                        recovered_neighbors = 0
                        for neighbor_index in network_graph.adj[node_index]:
                            if network_graph.nodes[neighbor_index]["state"] == "R":
                                recovered_neighbors += 1
                        if recovered_neighbors == network_graph.degree[node_index]:
                            state_change_flags[node_index] = 1

                for node_index in range(total_nodes):
                    if current_round == 0 or state_change_flags[node_index] == 0:
                        classification_models[node_index].train()
                        for features, labels in training_data_loaders[node_index]:
                            model_optimizers[node_index].zero_grad()
                            predictions = classification_models[node_index](features)
                            loss_value = loss_function(predictions, labels)
                            loss_value.backward()
                            model_optimizers[node_index].step()

                new_model_states = [dict() for index in range(total_nodes)]

                for node_index in range(total_nodes):
                    if network_graph.nodes[node_index]["state"] == "A":
                        participating_nodes = [node_index]
                        for neighbor_index in network_graph.adj[node_index]:
                            if network_graph.nodes[neighbor_index]["state"] == "A":
                                participating_nodes.append(neighbor_index)
                        weight_fraction = 1.0 / len(participating_nodes)

                        for name, parameter in classification_models[node_index].named_parameters():
                            new_model_states[node_index][name] = parameter.data.clone() * weight_fraction

                        for neighbor_index in participating_nodes[1:]:
                            for name, parameter in classification_models[neighbor_index].named_parameters():
                                new_model_states[node_index][name] += parameter.data.clone() * weight_fraction
                    else:
                        for name, parameter in classification_models[node_index].named_parameters():
                            new_model_states[node_index][name] = parameter.data.clone()

                for node_index in range(total_nodes):
                    for name, parameter in classification_models[node_index].named_parameters():
                        parameter.data.copy_(new_model_states[node_index][name])

                round_accuracies = []
                for node_index in range(total_nodes):
                    node_accuracy = evaluate_node_accuracy(classification_models[node_index], testing_data_loader)
                    round_accuracies.append(node_accuracy)
                    node_specific_accuracy[node_index].append(node_accuracy)

                global_accuracy_history.append(round_accuracies)

                # FIX 3: Communication cost calculation.
                #
                # Previous (WRONG): counted undirected A-A edges only.
                # This gave 0 whenever only 1 node was Active, even though
                # that node had neighbours it was "aware of" — and it also
                # ignored the bidirectional nature of model exchange.
                #
                # Correct logic:
                #   - Find all undirected edges where BOTH endpoints are A.
                #     These are the edges that actually carry parameter
                #     exchanges this round (as per the aggregation loop above).
                #   - Multiply by 2 because each such edge represents two
                #     directed transmissions: node i sends to j AND j sends to i.
                #   - If only 1 node is Active it has zero A-A edges → 0
                #     transmissions, which IS correct: a lone Active node
                #     aggregates with nobody and sends nothing.
                aa_undirected_edges = sum(
                    1 for i in range(total_nodes)
                    for j in range(i + 1, total_nodes)
                    if network_graph.has_edge(i, j)
                    and network_graph.nodes[i]["state"] == "A"
                    and network_graph.nodes[j]["state"] == "A"
                )
                # ×2 for bidirectional transmissions
                round_active_edges = aa_undirected_edges * 2

                active_edges_per_round.append(round_active_edges)
                nodes_active_per_round.append(population_counts[1])
                round_time_history.append(time.time() - start_time)

                average_round_accuracy = sum(round_accuracies) / total_nodes
                print(f"Round {current_round + 1:3d}/{communication_rounds} | "
                      f"Avg Accuracy: {average_round_accuracy * 100:.2f}% | "
                      f"Active Nodes: {population_counts[1]} | "
                      f"A-A edges (×2): {round_active_edges}")

                if current_round >= 0:
                    update_epidemic_states(network_graph, node_specific_accuracy, current_round, state_change_flags, saturation_tracker, saturation_time)

            # ── Post-run saving ────────────────────────────────────────────
            end_time = time.time()
            duration_seconds = end_time - start_time
            duration_minutes = duration_seconds / 60.0
            execution_durations.append(duration_minutes)

            # Model size in bytes (float32 = 4 bytes per parameter)
            model_param_count = sum(p.numel() for p in classification_models[0].parameters())
            model_size_bytes  = model_param_count * 4

            n_rounds         = len(global_accuracy_history)
            round_acc_avg    = [sum(r) / total_nodes for r in global_accuracy_history]
            round_acc_min    = [min(r) for r in global_accuracy_history]
            round_acc_max    = [max(r) for r in global_accuracy_history]
            round_acc_std    = [float(pandas.Series(r).std()) for r in global_accuracy_history]

            # bytes = directed transmissions × model size
            bytes_per_round  = [e * model_size_bytes for e in active_edges_per_round]
            cumulative_bytes = pandas.Series(bytes_per_round).cumsum()

            # ── per_round_metrics ──────────────────────────────────────────
            per_round_df = pandas.DataFrame({
                'round':               range(1, n_rounds + 1),
                'avg_accuracy':        round_acc_avg,
                'min_accuracy':        round_acc_min,
                'max_accuracy':        round_acc_max,
                'std_accuracy':        round_acc_std,
                'nodes_inactive':      [p[0] for p in population_counts_history[:n_rounds]],
                'nodes_active':        nodes_active_per_round[:n_rounds],
                'nodes_recovered':     [p[2] for p in population_counts_history[:n_rounds]],
                # directed transmissions this round (undirected AA edges × 2)
                'directed_transmissions': active_edges_per_round[:n_rounds],
                'bytes_transmitted':   bytes_per_round[:n_rounds],
                'cumulative_bytes':    cumulative_bytes.tolist(),
                'cumulative_MB':       cumulative_bytes.div(1024 ** 2).tolist(),
                'wall_clock_time_s':   round_time_history[:n_rounds],
            })
            per_round_df.to_excel(
                directory_path + "per_round_metrics_" + str(execution_index) + ".xlsx", index=False)

            # ── per_node_metrics ───────────────────────────────────────────
            node_rows = []
            for round_idx in range(n_rounds):
                for node_idx in range(total_nodes):
                    state = node_state_history[round_idx][node_idx]
                    active_nbrs = sum(
                        1 for nb in network_graph.neighbors(node_idx)
                        if node_state_history[round_idx][nb] == 'A'
                    )
                    node_rows.append({
                        'round':             round_idx + 1,
                        'node_id':           node_idx,
                        'accuracy':          global_accuracy_history[round_idx][node_idx],
                        'state':             state,
                        'is_training':       1 if state in ('I', 'A') else 0,
                        'is_aggregating':    1 if state == 'A' else 0,
                        'num_neighbours':    network_graph.degree[node_idx],
                        'active_neighbours': active_nbrs,
                        # directed sends from this node this round
                        'sends_this_round':  active_nbrs if state == 'A' else 0,
                    })
            pandas.DataFrame(node_rows).to_excel(
                directory_path + "per_node_metrics_" + str(execution_index) + ".xlsx", index=False)

            # ── comm_cost_analysis ─────────────────────────────────────────
            comm_cost_df = pandas.DataFrame({
                'round':                  range(1, n_rounds + 1),
                'nodes_active':           nodes_active_per_round[:n_rounds],
                'participation_rate':     [a / total_nodes for a in nodes_active_per_round[:n_rounds]],
                # undirected AA edges (what exists in the graph)
                'aa_undirected_edges':    [e // 2 for e in active_edges_per_round[:n_rounds]],
                # directed transmissions (what actually gets sent = ×2)
                'directed_transmissions': active_edges_per_round[:n_rounds],
                'bytes_this_round':       bytes_per_round[:n_rounds],
                'cumulative_bytes':       cumulative_bytes.tolist(),
                'cumulative_MB':          cumulative_bytes.div(1024 ** 2).tolist(),
                'accuracy_per_MB':        [
                                              round_acc_avg[i] / max(1e-9, cumulative_bytes.iloc[i] / (1024 ** 2))
                                              for i in range(n_rounds)
                                          ],
            })
            comm_cost_df.to_excel(
                directory_path + "comm_cost_analysis_" + str(execution_index) + ".xlsx", index=False)

            # ── summary ────────────────────────────────────────────────────
            total_bytes  = int(cumulative_bytes.iloc[-1]) if n_rounds > 0 else 0
            final_acc    = round_acc_avg[-1] if round_acc_avg else 0
            peak_acc     = max(round_acc_avg) if round_acc_avg else 0
            threshold    = 0.90 * peak_acc
            rounds_to_90 = next((i + 1 for i, a in enumerate(round_acc_avg) if a >= threshold), n_rounds)

            ia_trans = sum(
                1 for r in range(1, len(node_state_history))
                for n in range(total_nodes)
                if node_state_history[r-1][n] == 'I' and node_state_history[r][n] == 'A'
            )
            ar_trans = sum(
                1 for r in range(1, len(node_state_history))
                for n in range(total_nodes)
                if node_state_history[r-1][n] == 'A' and node_state_history[r][n] == 'R'
            )
            ra_trans = sum(
                1 for r in range(1, len(node_state_history))
                for n in range(total_nodes)
                if node_state_history[r-1][n] == 'R' and node_state_history[r][n] == 'A'
            )

            pandas.DataFrame([{
                'method':                          'IADA-NET-D-DFL',
                'topology':                        'Watts-Strogatz',
                'avg_degree_k':                    graph_parameter,
                'rewiring_prob_p':                 0.3,
                'dataset':                         'MNIST',
                'execution_index':                 execution_index,
                'total_nodes':                     total_nodes,
                'communication_rounds_run':        n_rounds,
                'final_accuracy':                  final_acc,
                'best_accuracy':                   peak_acc,
                'total_time_s':                    duration_seconds,
                'total_time_min':                  duration_minutes,
                'total_directed_transmissions':    sum(active_edges_per_round[:n_rounds]),
                'total_bytes_transmitted':         total_bytes,
                'total_bytes_MB':                  total_bytes / (1024 ** 2),
                'avg_active_nodes_per_round':      sum(nodes_active_per_round[:n_rounds]) / max(1, n_rounds),
                'avg_directed_transmissions_per_round': sum(active_edges_per_round[:n_rounds]) / max(1, n_rounds),
                'avg_participation_rate':          sum(nodes_active_per_round[:n_rounds]) / max(1, n_rounds * total_nodes),
                'rounds_to_90pct_peak':            rounds_to_90,
                'total_IA_transitions':            ia_trans,
                'total_AR_transitions':            ar_trans,
                'total_RA_transitions':            ra_trans,
                'model_param_count':               model_param_count,
                'model_size_bytes':                model_size_bytes,
                'seed_node':                       maximum_degree_node,
                'initial_graph_edges':             network_graph.number_of_edges(),
            }]).to_excel(
                directory_path + "summary_" + str(execution_index) + ".xlsx", index=False)

            # ── Original files (kept unchanged) ───────────────────────────
            pandas.DataFrame(global_accuracy_history).to_excel(
                directory_path + "all_acc_client" + str(execution_index) + ".xlsx", index=False)
            pandas.DataFrame(node_specific_accuracy).to_excel(
                directory_path + "accall" + str(execution_index) + ".xlsx", index=False)
            pandas.DataFrame(population_counts_history).to_excel(
                directory_path + "IA_list" + str(execution_index) + ".xlsx", index=False)
            pandas.DataFrame(node_state_history).to_excel(
                directory_path + "IA_state" + str(execution_index) + ".xlsx", index=False)

            print(f"\n  Run {execution_index} | Acc: {final_acc*100:.2f}% | "
                  f"Total comm: {total_bytes/(1024**2):.2f} MB | "
                  f"I→A: {ia_trans}  A→R: {ar_trans}  R→A: {ra_trans}")

            average_accuracy_curve = [sum(round_data) / total_nodes for round_data in global_accuracy_history]

            figure_object = plotting_module.figure(figsize=(14, 6))

            accuracy_axis = plotting_module.subplot(1, 2, 1)
            accuracy_axis.plot(range(len(average_accuracy_curve)), average_accuracy_curve,
                               color='red', marker='o', label='Training Accuracy')
            accuracy_axis.set_xlabel('Communication Rounds')
            accuracy_axis.set_ylabel('Average Test Accuracy')
            accuracy_axis.set_title('Convergence Curve')
            accuracy_axis.grid(True)
            accuracy_axis.legend()

            topology_axis = plotting_module.subplot(1, 2, 2)
            node_color_map = []
            for node_index in network_graph.nodes():
                if network_graph.nodes[node_index]["state"] == "I":
                    node_color_map.append('blue')
                elif network_graph.nodes[node_index]["state"] == "A":
                    node_color_map.append('red')
                else:
                    node_color_map.append('green')

            layout_positions = graph_library.spring_layout(network_graph, seed=42)
            graph_library.draw_networkx_nodes(network_graph, layout_positions,
                                              node_size=200, node_color=node_color_map, ax=topology_axis)
            graph_library.draw_networkx_edges(network_graph, layout_positions,
                                              alpha=0.5, ax=topology_axis)
            topology_axis.set_title('Final Network Topology (I=Blue, A=Red, R=Green)')
            topology_axis.axis('off')

            plotting_module.tight_layout()
            plotting_module.savefig(directory_path + "convergence_plot.png")
            plotting_module.show()

        duration_dataframe = pandas.DataFrame(execution_durations)
        duration_dataframe.to_excel(directory_path + "T.xlsx", index=False)

execute_net_d_dfl_simulation()