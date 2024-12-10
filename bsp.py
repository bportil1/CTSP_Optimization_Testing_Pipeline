import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys 

sys.setrecursionlimit(9999999999999)

class BSPNode:
    def __init__(self, bounds, axis=None, split_value=None, left=None, right=None, fitness=None, parent=None):
        self.bounds = bounds  # The bounding box for this node (a list of min and max for each dimension)
        self.axis = axis  # The axis along which the split is done
        self.split_value = split_value  # The value at which the space is split (midpoint)
        self.left = left  # Left child
        self.right = right  # Right child
        self.fitness = fitness  # Random fitness value assigned to this node
        self.parent = parent  # Parent node (for updating relationships)

    '''
    def update_fitness(self, new_fitness):
        """Method to update the fitness of the node dynamically."""
        # Fitness calculation can be customized, using volume or other properties
        self.fitness = new_fitness
        if self.parent:
            self.parent.update_fitness(new_fitness)  
    '''

    def update_fitness(self, new_fitness):
        # Using an iterative approach to update fitness up the tree
        current_node = self
        while current_node is not None:
            # Update fitness for the current node
            current_node.fitness = new_fitness
            # Move to the parent node
            current_node = current_node.parent

    def split(self, new_fitness):
        """Split the current region into two new regions."""
        # Randomly choose an axis for the split
        self.axis = random.randint(0, len(self.bounds) - 1)  # Choose axis (0 to n-1)
        
        # Calculate the midpoint for the split along the chosen axis
        split_value = (self.bounds[self.axis][0] + self.bounds[self.axis][1]) / 2
        self.split_value = split_value

        # Split bounds into two parts
        left_bounds = [list(dim) for dim in self.bounds]
        right_bounds = [list(dim) for dim in self.bounds]
        
        left_bounds[self.axis][1] = split_value  # Update the max value along the split axis
        right_bounds[self.axis][0] = split_value  # Update the min value along the split axis
        
        # Create left and right children with the correct parent reference
        self.left = BSPNode(left_bounds, fitness=random.random(), parent=self)
        self.right = BSPNode(right_bounds, fitness=random.random(), parent=self)
        
        # Update the fitness of the current node after splitting
        self.update_fitness(new_fitness)

def generate_bsp_tree(bounds, depth, max_depth):
    if depth > max_depth:
        return None

    # Randomly choose an axis for the split
    axis = random.randint(0, len(bounds) - 1)  # Choose axis (0 to n-1)

    #print("bounds: ", bounds)

    # Calculate the midpoint along the chosen axis
    min_val, max_val = bounds[axis][0], bounds[axis][1]
    split_value = (min_val + max_val) / 2
    
    # Split the bounds into two halves
    left_bounds = [list(dim) for dim in bounds]
    left_bounds[axis][1] = split_value  # Update the max value along the split axis
    
    right_bounds = [list(dim) for dim in bounds]
    right_bounds[axis][0] = split_value  # Update the min value along the split axis
    
    # Recursively generate left and right children
    #fitness = random.random()
    fitness = float('inf')
    left_node = generate_bsp_tree(left_bounds, depth + 1, max_depth)
    right_node = generate_bsp_tree(right_bounds, depth + 1, max_depth)
    
    return BSPNode(bounds, axis, split_value, left_node, right_node, fitness)

def find_region(node, point):
    """Finds the region to which the given point belongs using an iterative approach."""
    current_node = node
    while current_node is not None:
        # Check if the point is inside the current node's bounds
        inside = True
        for i, (min_val, max_val) in enumerate(current_node.bounds):
            if not (min_val <= point[i] <= max_val):
                inside = False
                break
        
        if inside:
            # If it's a leaf node, return this node
            if current_node.left is None and current_node.right is None:
                return current_node

            # Debugging print statements to track the splits and decisions
            #print(f"Point: {point.tolist()}, Axis: {current_node.axis}, Split Value: {current_node.split_value}")

            # Determine which child to traverse based on the split value
            if point[current_node.axis] < current_node.split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        else:
            return None

    return None


'''
def find_region(node, point):
    """Recursively finds the region to which the given point belongs."""
    if node is None:
        return None
    
    # Check if the point is inside the current node's bounds
    inside = True
    for i, (min_val, max_val) in enumerate(node.bounds):
        if not (min_val <= point[i] <= max_val):
            inside = False
            break
    
    if inside:
        # If the point is inside the bounds, return the current node directly
        if node.left is None and node.right is None:
            return node  # This is a leaf node, so return this node
     
        #print(f"Point: {point}, Axis: {node.axis}, Split Value: {node.split_value}")
        
        # If not a leaf, we will go deeper into the tree depending on the axis
        if point[node.axis] < node.split_value:
            if node.left:
                return find_region(node.left, point)
        else:
            if node.right:
            	return find_region(node.right, point)
    
    return None
'''

def find_neighbors(node, n):
    """Finds neighboring regions of a given node and returns them."""
    neighbors = []
    #print("Node: ", node.left)
    # Check if the node has left and right children and add them as neighbors
    if node is not None:
        if node.left is not None: 
            neighbors.append(node.left)
        if node.right is not None:
            neighbors.append(node.right)
    # Limit the number of neighbors to 'n'
    return neighbors[:n]

def find_min_fitness_region(node, neighbors, point):
    """Finds the region with the minimum fitness value from neighbors and checks if the point is inside it."""
    min_fitness = float('inf')
    best_region = None
    
    if neighbors == []:
        best_region = node

    for neighbor in neighbors:
        if neighbor.fitness < min_fitness and is_point_in_region(neighbor, point):
            min_fitness = neighbor.fitness
            best_region = neighbor
    
    return best_region

def is_point_in_region(region, point):
    """Checks if the point lies within the given region's bounds."""
    for i, (min_val, max_val) in enumerate(region.bounds):
        if not (min_val <= point[i] <= max_val):
            return False
    return True

def find_min_fitness_region_recursive(node):
    """Recursively traverses the BSP tree and finds the region with the minimum fitness value."""
    if node is None:
        return None
    
    # Initialize the best region as the current node
    min_fitness_region = node
    
    # Traverse the left subtree if it exists
    if node.left:
        left_region = find_min_fitness_region_recursive(node.left)
        if left_region and left_region.fitness < min_fitness_region.fitness:
            min_fitness_region = left_region
    
    # Traverse the right subtree if it exists
    if node.right:
        right_region = find_min_fitness_region_recursive(node.right)
        if right_region and right_region.fitness < min_fitness_region.fitness:
            min_fitness_region = right_region
    
    # Return the region with the minimum fitness value
    return min_fitness_region

def find_bounding_box_center(bounds):
    """Finds the center of a bounding box in n dimensions.
    
    Args:
        bounds (list): A list of tuples representing the bounding box for each dimension [(min, max), ...]
        
    Returns:
        list: A list representing the center of the bounding box in each dimension [center_x, center_y, ..., center_n]
    """
    center = []
    for min_val, max_val in bounds:
        center.append((min_val + max_val) / 2)
    return center

