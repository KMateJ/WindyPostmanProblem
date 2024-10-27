import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
import copy
import random
import importlib


def generate_random_graph(node_count, connectivity):
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(1, node_count + 1))

    # Ensure at least min_edges are added for connectivity

    # Add edges to achieve the desired connectivity
    while len(list(nx.isolates(G))) >0:
        node1 = random.randint(1, node_count)
        node2 = random.randint(1, node_count)
        if node1 != node2 and not G.has_edge(node1, node2):
            G.add_edge(node1, node2)

    # Add remaining edges randomly
    remaining_edges = int(node_count * (node_count - 1) * connectivity / 2)
    for _ in range(remaining_edges):
        node1 = random.randint(1, node_count)
        node2 = random.randint(1, node_count)
        if node1 != node2 and not G.has_edge(node1, node2):
            G.add_edge(node1, node2)

    return G

def random_walk(graph):
    visited_edges = set()
    walked_edges = []
    nodes = list(graph.nodes())
    
    current_node = random.choice(nodes)
    while len(visited_edges) < graph.number_of_edges():
        
        neighbors = list(graph.neighbors(current_node))
        next_node = random.choice(neighbors)
        walked_edge = (current_node, next_node)
        edge = (min(current_node, next_node), max(current_node, next_node))

        visited_edges.add(edge)
        
        walked_edges.append(walked_edge)
        #print(f"Visited edge: {walked_edge}")
        current_node = next_node

    return walked_edges

def visualize_graph(G, title="Graph Visualization", seed=42):
    plt.figure(figsize=(18,8), facecolor='#5e5e5e')
    ax = plt.gca()
    ax.set_facecolor('#5e5e5e')
    pos = nx.spring_layout(G, weight=None, seed=seed)  # You can use other layouts as well

    # Check if edges have weights
    if nx.get_edge_attributes(G, 'weight'):
        # If weights exist, use them for edge coloring
        edge_weights = np.array([G[u][v].get('weight', 1) for u, v in G.edges()])
        cmap = plt.cm.plasma
        norm_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())
        min_alpha = 0.1  # Minimum alpha for visibility
        alphas = norm_weights * (1 - min_alpha) + min_alpha
        edge_colors = [cmap(weight) for weight in norm_weights]
        edge_colors_with_alpha = [(r, g, b, a) for (r, g, b, _), a in zip(edge_colors, alphas)] # Use weights for edge coloring
    else:
        # If no weights, set all edges to the same color (e.g., gray)
        edge_colors_with_alpha = 'blue'
        edge_weights = None  # No weights present

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrowsize=10,
        edge_color=edge_colors_with_alpha,
        width=2,
    )

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=min(5000/G.number_of_nodes(), 200))
    
    # If edge weights exist, create colorbar based on scalar values (not RGBA tuples)
    if edge_weights is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max()))
        sm.set_array([])  # Create an empty array for the ScalarMappable
        plt.colorbar(sm, label='Edge Weight', orientation='vertical', fraction=0.046, pad=0.04)
    
    plt.title(title)
    plt.show()
    
def add_weights_to_edges_from_dict(graph, weights_dict):
    for edge, weight in weights_dict.items():
        if graph.has_edge(*edge):
            graph[edge[0]][edge[1]]['weight'] = (1/max(weight,0.0001))
        elif graph.has_edge(*reversed(edge)):
            graph[reversed(edge)[0]][reversed(edge)[1]]['weight'] = (1/max(weight,0.0001))
        else:
            print(f"Edge {edge} not found in the graph.")
            
def probabilistic_walk(graph):
    visited_edges = set()
    walked_edges = []
    nodes = list(graph.nodes())
    
    current_node = random.choice(nodes)
    while len(visited_edges) < graph.number_of_edges():
        neighbors = list(graph.neighbors(current_node))
        weights = [graph[current_node][neighbor].get('weight', 1) for neighbor in neighbors]
        if len(weights) == 0:
            print("some fuckery")

        # Normalize weights to create probabilities
        try:
            probabilities = [weight / sum(weights) for weight in weights]
        except:
            print(weights)
            return
            

        next_node = random.choices(neighbors, weights=probabilities)[0]
        edge = (min(current_node, next_node), max(current_node, next_node))

        visited_edges.add(edge)
        walked_edges.append(edge)
        #print(f"Visited edge: {edge}")
        
        graph[current_node][next_node]['weight'] = np.sqrt(graph[current_node][next_node]['weight'])
        
        
        current_node = next_node

    return walked_edges

def generate_sf_graph(n):
    G = nx.scale_free_graph(n=n,alpha=.5,beta=.25,gamma=.25).to_undirected()
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G = nx.Graph(G)
    G = nx.convert_node_labels_to_integers(G)
    components = list(nx.connected_components(G))
    while len(components) > 1:
        G.add_edge(components[0].pop(), components[1].pop())
        components = list(nx.connected_components(G))
    return G


def get_highest_degree_node(graph):
    """Return the node with the highest degree in the graph."""
    degrees = dict(graph.degree())
    return max(degrees, key=degrees.get)

def initialize_edge_weights(graph):
    """Set all edge weights to 0 for visualization."""
    for u, v in graph.edges():
        graph[u][v]['weight'] = 0


def multiple_probabilistic_walks(graph, num_walks, steps):
    """Simulate multiple random walks on the weighted graph."""
    walked_edges_per_step = [[] for _ in range(steps)]  # Store walked edges per step
    edge_step_count = {edge: 0 for edge in graph.edges()}  # Track how many times each edge is stepped on
    starting_node = get_highest_degree_node(graph)  # Start walks from the highest degree node

    for _ in range(num_walks):
        current_node = starting_node
        
        for i in range(steps):
            neighbors = list(graph.neighbors(current_node))
            weights = [graph[current_node][neighbor].get('weight', 1) for neighbor in neighbors]

            # Normalize weights to create probabilities
            probabilities = [weight / sum(weights) if sum(weights) != 0 else 1 / len(weights) for weight in weights]
            next_node = random.choices(neighbors, weights=probabilities)[0]
            edge = (min(current_node, next_node), max(current_node, next_node))

            walked_edges_per_step[i].append(edge)
            edge_step_count[edge] += 1  # Update step count for the edge


            current_node = next_node

    v_graph = copy.deepcopy(graph)
    print(v_graph)
    initialize_edge_weights(v_graph)
    snapshots = []
    for  edges_in_step in walked_edges_per_step:
        for edge in edges_in_step:
            u, v = edge
            v_graph[u][v]['weight'] += 1 
        snapshot = copy.deepcopy(v_graph)
        snapshots.append(snapshot)
        
    for i in range (90):
        snapshot = copy.deepcopy(snapshots[-1])
        snapshots.append(snapshot)
    
    return snapshots 


def compute_max_weights(snapshots):
    """Compute the maximum weight for each edge across all snapshots."""
    max_weights = {}
    for snapshot in snapshots:
        for u, v, data in snapshot.edges(data=True):
            edge = (min(u, v), max(u, v))
            weight = data.get('weight', 0)
            if edge not in max_weights:
                max_weights[edge] = weight
            else:
                max_weights[edge] = max(max_weights[edge], weight)
    return max_weights


def visualize_graph(snapshot, ax, max_weight, starting_node=None):
    """Visualize a single snapshot of the graph with edge opacity and color based on current weight."""
    ax.clear()  # Clear the previous frame
    pos = nx.spring_layout(snapshot, seed=42, weight=None)  # Fixed layout for consistency
    
    # Get edge weights for the current snapshot
    edge_weights = np.array([snapshot[u][v]['weight'] for u, v in snapshot.edges()])

    # Create a colormap for edge weights
    cmap = plt.cm.plasma
    min_weight = 0.1  # Minimum weight to ensure edges are visible
    weight_factor = 3  # Adjust this factor to enhance color intensity
    
    # Minimum and maximum opacity values
    min_opacity = 0.1  # Set a minimum opacity for visibility
    max_opacity = 1.0  # Fully opaque for the highest weights

    # Scale weights for color mapping
    edge_weights = np.maximum(edge_weights, min_weight)  # Avoid weights below minimum
    
    # Normalize edge weights for color and opacity calculation
    scaled_weights = edge_weights * weight_factor  # Scale weights for color mapping
    
    # Calculate opacity based on edge weights relative to max weight in the last snapshot
    opacities = min_opacity + (edge_weights / max_weight) * (max_opacity - min_opacity)
    
    # Get colors from the colormap and apply opacity
    edge_colors = [(*cmap(weight / max_weight)[:3], opacity) 
                   for weight, opacity in zip(scaled_weights, opacities)]

    # Draw edges with colors and transparency based on weights
    nx.draw_networkx_edges(snapshot, pos, edge_color=edge_colors, width=2, ax=ax)

    # Draw nodes, differentiating the starting node with a different color
    node_colors = ['red' if node == starting_node else 'skyblue' for node in snapshot.nodes()]
    nx.draw_networkx_nodes(snapshot, pos, node_color=node_colors, node_size=100, ax=ax)

    # Turn off the axis
    plt.axis('off')


def animate_walks(snapshots, save_gif=True, filename="multiple_walks.gif", fps=5):
    """Animate the snapshots of graph walks and save as a GIF if desired."""
    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(18, 8), facecolor='#3e3e3e')
    last_snapshot = snapshots[-1]
    max_weight = max([last_snapshot[u][v]['weight'] for u, v in last_snapshot.edges()])
    starting_node = get_highest_degree_node(snapshots[0]) 
    
    # Update function to visualize each snapshot
    def update(frame):
        ax.clear()  # Clear the previous frame
        visualize_graph(snapshots[frame], ax, max_weight,starting_node)  # Visualize the current snapshot
        plt.title(f"Frame {frame + 1}", color='white')  # Update title with current step number
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(snapshots), interval=300, repeat=False)

    # Show or save the animation
    if save_gif:
        writer = PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
        print(f"Animation saved as {filename}")
    else:
        plt.show()