import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from optimizers_tsp import *
import pandas as pd
import itertools
import time
import plotly.graph_objects as go
import os

class ContinuousTSP:
    def __init__(self, cities):
        self.cities = np.array(cities)
        self.num_cities = len(cities)

    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def total_distance(self, tour):
        dist = 0
        for i in range(len(tour) - 1):
            dist += self.euclidean_distance(self.cities[tour[i]], self.cities[tour[i+1]])
        dist += self.euclidean_distance(self.cities[tour[-1]], self.cities[tour[0]])  # Return to the starting point
        return dist


    def gradient_computation(self, tour):
        gradient = np.zeros_like(self.cities, dtype=np.float64)
        # Calculate the pairwise Euclidean distances
        dist_matrix = cdist(self.cities[tour], self.cities[tour])
        # Compute the gradients for each pair of cities
        for i in range(self.num_cities - 1):
            for j in range(i + 1, self.num_cities):
                diff = self.cities[tour[i]] - self.cities[tour[j]]
                gradient[tour[i]] -= diff / np.linalg.norm(diff)
                gradient[tour[j]] += diff / np.linalg.norm(diff)
        diff = self.cities[tour[-1]] - self.cities[tour[0]]
        gradient[tour[-1]] -= diff / np.linalg.norm(diff)
        gradient[tour[0]] += diff / np.linalg.norm(diff)        
        #print("gradient: ", gradient)
        return np.sum(gradient)

def sba_test(tsp, filename, run_num):

    sba = SwarmBasedAnnealingOptimizer(tsp, 10, 3, 10)

    start_time = time.time()
    best_tour, best_distance, obj_ct = sba.optimize()
    exc_time = time.time() - start_time

    data = {'best_tour': [str(best_tour)], 'best_distance': [best_distance], 'obj_exc_ct': [obj_ct], 'runtime': [exc_time]}

    print("SBA Best tour:", best_tour)
    print("SBA Best distance:", best_distance)

    df = pd.DataFrame(data)

    #df = pd.read_csv(filename)
    #new_row = pd.DataFrame([row], columns=df.columns)  # Create a DataFrame for the new row
    #df = pd.concat([df, new_row], ignore_index=True)  # Concatenate the new row
    df.to_csv(filename, index=False)        

    cities = tsp.cities  # Assuming tsp.cities gives the coordinates of the cities
    closed_tour = np.append(best_tour, best_tour[0])  # Close the tour by adding the first city at the end

    # Create a Plotly figure
    fig = go.Figure()

    # Plot the cities as red markers
    fig.add_trace(go.Scatter3d(
        x=cities[:, 0],
        y=cities[:, 1],
        z=cities[:, 2],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Cities'
    ))

    # Plot the optimal tour path in blue
    fig.add_trace(go.Scatter3d(
        x=cities[closed_tour, 0],
        y=cities[closed_tour, 1],
        z=cities[closed_tour, 2],
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=6, color='blue'),
        name='Optimal Tour'
    ))

    # Update layout with titles and labels
    fig.update_layout(
        title="Optimal Tour for SBA and Continuous TSP",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=True
    )

    plot = os.path.join("./optimizer_results/sba_results", f"sba_plot_{run_num}.html")
    os.makedirs(os.path.dirname(plot), exist_ok=True)
    fig.write_html(plot)


def hdffa_test(tsp, filename, run_num):

    ######### HD FIREFLY

    hdffsa = HdFireflySimulatedAnnealingOptimizer(tsp, 3, [0, 110], pop_test=10)

    start_time = time.time()
    best_tour, best_distance, obj_ct = hdffsa.optimize()
    exc_time = time.time() - start_time

    data = {'best_tour': [str(best_tour)], 'best_distance': [best_distance], 'obj_exc_ct': [obj_ct], 'runtime': [exc_time]}

    print("HDFFA Best tour:", best_tour)
    print("HDFFA Best distance:", best_distance)

    df = pd.DataFrame(data)

    #df = pd.read_csv(filename)
    #new_row = pd.DataFrame([row], columns=df.columns)  # Create a DataFrame for the new row
    #df = pd.concat([df, new_row], ignore_index=True)  # Concatenate the new row
    df.to_csv(filename, index=False) 

    cities = tsp.cities  # Assuming tsp.cities gives the coordinates of the cities
    closed_tour = np.append(best_tour, best_tour[0])  # Close the tour by adding the first city at the end

    # Create a Plotly figure
    fig = go.Figure()

    # Plot the cities as red markers
    fig.add_trace(go.Scatter3d(
        x=cities[:, 0],
        y=cities[:, 1],
        z=cities[:, 2],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Cities'
    ))

    # Plot the optimal tour path in blue
    fig.add_trace(go.Scatter3d(
        x=cities[closed_tour, 0],
        y=cities[closed_tour, 1],
        z=cities[closed_tour, 2],
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=6, color='blue'),
        name='Optimal Tour'
    ))

    # Update layout with titles and labels
    fig.update_layout(
        title="Optimal Tour for HDFFA and Continuous TSP",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=True
    )

    plot = os.path.join("./optimizer_results/hdffa_results", f"hdffa_plot_{run_num}.html")
    os.makedirs(os.path.dirname(plot), exist_ok=True)
    fig.write_html(plot)


def bf_test(tsp, filename, run_num):
    obj_ct = 0
    # Generate all permutations of the points
    permutations = itertools.permutations(range(tsp.num_cities))

    # Find the shortest tour
    shortest_tour = None
    min_distance = float('inf')
    start_time = time.time()
    for perm in permutations:
        perm_distance = tsp.total_distance(perm)
        obj_ct += 1
        if perm_distance < min_distance:
            min_distance = perm_distance
            shortest_tour = np.array(perm)


    idx_tour = shortest_tour
    shortest_tour = np.array([tsp.cities[tour] for tour in shortest_tour])
    exc_time = time.time() - start_time

    # Output the results
    print("BF Shortest tour:", idx_tour)
    print("BF Minimum distance:", min_distance)

    data = {'best_tour': [str(idx_tour)], 'best_distance': [min_distance], 'obj_exc_ct': [obj_ct], 'runtime': [exc_time]}

    df = pd.DataFrame(data)
    
    #df = pd.read_csv(filename)
    #new_row = pd.DataFrame([row], columns=df.columns)  # Create a DataFrame for the new row
    #df = pd.concat([df, new_row], ignore_index=True)  # Concatenate the new row
    df.to_csv(filename, index=False) 

    x = [point[0] for point in shortest_tour] + [shortest_tour[0][0]]  # Add the first point at the end to close the loop
    y = [point[1] for point in shortest_tour] + [shortest_tour[0][1]]
    z = [point[2] for point in shortest_tour] + [shortest_tour[0][2]]

    fig = go.Figure()

    # Plot the points (cities)
    fig.add_trace(go.Scatter3d(
        x=tsp.cities[:, 0], 
        y=tsp.cities[:, 1], 
        z=tsp.cities[:, 2],
        mode='markers', 
        marker=dict(size=5, color='red'),
        name='Cities'
    ))

    # Plot the shortest cycle (tour)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers', 
        line=dict(color='blue', width=3),
        marker=dict(size=6, color='blue'),
        name='Shortest Tour'
    ))

    # Update layout (labels and title)
    fig.update_layout(
        title="Optimal Tour for BF and Continuous TSP",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=True
    )

    plot = os.path.join("./optimizer_results/bf_results", f"bf_plot_{run_num}.html")
    os.makedirs(os.path.dirname(plot), exist_ok=True)
    fig.write_html(plot)

def random_city_tsp_generator(num_cities):

    n = num_cities  # Number of cities
    coordinate_range = 100  # Range for each coordinate (0 to 9)

    # Generate random cities with 3 coordinates each
    cities = np.random.randint(0, coordinate_range, size=(n, 3))

    tsp = ContinuousTSP(cities)

    return tsp


def test_runner():
    
    filename_base = './optimizer_results/'

    sba_dir = filename_base + 'sba_results/'
    os.makedirs(sba_dir, exist_ok=True)  
    hdffa_dir = filename_base + 'hdffa_results/'
    os.makedirs(hdffa_dir, exist_ok=True)  
    bd_dir = filename_base + 'bf_results/'
    os.makedirs(bd_dir, exist_ok=True)  

    num_cities = [2] #5,10,15]
    
    for idx1 in num_cities:
        for idx2 in range(2):
            tsp = random_city_tsp_generator(idx1)
            filename = filename_base + 'sba_results/sba_run_' + str(idx2) + '_' + str(idx1) + '_cities.csv'
            sba_test(tsp, filename, idx2)
            filename = filename_base + 'hdffa_results/hdffa_run_' + str(idx2) + '_' + str(idx1) + '_cities.csv'
            hdffa_test(tsp, filename, idx2)            
            filename = filename_base + 'bf_results/bf_run_' + str(idx2) + '_' + str(idx1) + '_cities.csv'          
            bf_test(tsp, filename, idx2)


if __name__ == '__main__':
    test_runner()           






