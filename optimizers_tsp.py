import numpy as np
from bsp import *
import math
import plotly.express as px
import plotly.graph_objects as go

class SimulatedAnnealingOptimizer:
    def __init__(self, ctsp_obj, curr_pt, init_error=None, max_iterations=1000, temperature=10, min_temp=.001, cooling_rate=.90, k=1, obj_comp_temp = 0):
        self.ctsp_obj = ctsp_obj
        self.curr_pt = curr_pt        
        self.init_error = init_error

        self.temperature = temperature
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        
        self.obj_comp_temp = obj_comp_temp

    def optimize(self):
        print("Beggining Simulated Annealing Optimization")
        update_ctr = 0
        print("Initial Tour: ", self.curr_pt)
        if self.init_error != None:
            curr_energy = self.init_error
        else:
            curr_energy = self.ctsp_obj.total_distance(self.curr_pt)
            self.obj_comp_temp += 1
        print("Initial Error: ", curr_energy)
        for idx in range(self.max_iterations):
            new_position = self.solution_transition(self.curr_pt)
            new_energy = self.ctsp_obj.total_distance(new_position)
            self.obj_comp_temp += 1
            print("Potential New Postition: ", new_position)
            print("Potential New Postition Error: ", new_energy)

            alpha = self.acceptance_probability_computation(curr_energy, new_energy)
            
            if new_energy < curr_energy: 
                self.curr_pt = new_position
                curr_energy = new_energy
                update_ctr = 0
            elif np.random.rand() > (1-alpha):            
                self.curr_pt = new_position
                curr_energy = new_energy
                update_ctr = 0
            else:
                update_ctr += 1
            self.temperature *= self.cooling_rate

            print("Current Position: ", self.curr_pt)
            print("Current Error: ", curr_energy)
            print("Current Temperature: ", self.temperature)

            if self.temperature < self.min_temp or update_ctr == 10:
                print("Early Convergence, Breaking")
                break
        
        print("Final Error: ", curr_energy)
        print("Final Position: ", self.curr_pt)
        return self.curr_pt, curr_energy, self.obj_comp_temp 

    def solution_transition(self, tour):
        new_position = tour.copy()
        i, j = random.sample(range(len(tour)), 2)
        new_position[i], new_position[j] = new_position[j], new_position[i]
        return new_position

    def acceptance_probability_computation(self, curr_energy, new_energy):
        if new_energy < curr_energy:
            return 1.0
        elif new_energy == curr_energy:
            return 0
        else:
            return np.exp(-((new_energy - curr_energy) / (self.temperature*.8)) )

class SwarmBasedAnnealingOptimizer:
    def __init__(self, ctsp_obj, num_particles, dimensions, max_iter, h=0.99): 
        self.ctsp_obj =  ctsp_obj

        self.num_particles = num_particles         
        self.dimensions = dimensions               
        self.max_iter = max_iter                   
        self.h = h                                 
                                                               
        self.paths = [[] for _ in range(self.num_particles)]
        self.values = [[] for _ in range(self.num_particles)]

        self.provisional_minimum = float('inf')

        self.positions = np.array([np.random.permutation(self.ctsp_obj.num_cities) for _ in range(self.num_particles)])

        self.masses = np.ones((1, self.num_particles))[0] * (1/self.num_particles)

        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([ self.ctsp_obj.total_distance(self.positions[p]) for p in range(self.num_particles)])

        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)

        self.obj_comp_temp = self.num_particles

    def update_mass(self, particle_idx):
        fitness = self.ctsp_obj.total_distance(self.positions[particle_idx])
        self.obj_comp_temp += 1
        new_mass = self.masses[particle_idx] - (self.h*(fitness - self.provisional_minimum)*self.masses[particle_idx])
        new_mass = np.clip(new_mass, 1e-6, 1)
        return new_mass
    
    def update_position(self, particle_idx, eta, iteration):
        gradient = self.ctsp_obj.gradient_computation(self.positions[particle_idx])
        inv_mass = np.mean(self.masses)
        new_position = self.positions[particle_idx] - (self.h*gradient*self.ctsp_obj.total_distance(self.positions[particle_idx])) + (np.sqrt(2*self.h*inv_mass)*eta)
        self.obj_comp_temp += 1
        new_position = np.argsort(new_position) 
        return new_position

    def provisional_min_computation(self):
        self.obj_comp_temp += self.num_particles
        return np.sum(self.masses * np.array([self.ctsp_obj.total_distance(self.positions[y]) for y in range(self.num_particles)])) / np.sum(self.masses)

    def optimize(self):
        print("Beggining Optimization")
        self.provisional_minimum = self.provisional_min_computation()
        print("Provisional Minimum: ", self.provisional_minimum)
        
        for iteration in range(self.max_iter):
            print("Current Iteration: ", iteration)
            for i in range(self.num_particles):
                # Update velocity
                new_mass = self.update_mass(i)
                self.masses[i] = new_mass

            h = self.h * np.exp(-.99 * iteration)
            eta = np.random.normal(0, 1, (self.num_particles, self.ctsp_obj.num_cities)) * np.exp(-.99*iteration)

            for i in range(self.num_particles):

                print("Initial Position for Agent ", i, ":", self.positions[i])

                new_position = self.update_position(i, eta[i], iteration)
                    
                self.positions[i] = new_position

                print("Current Position for Agent ", i, ":", new_position)

                # Evaluate new fitness
                fitness = self.ctsp_obj.total_distance(new_position)
                self.obj_comp_temp += 1
                self.paths[i].append(self.positions[i].copy())
                self.values[i].append(fitness)

                print("Current Fitness for Agent ", i, ":", fitness)

                # Update personal best if necessary
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                min_fitness_idx = np.argmin(self.personal_best_fitness)
                if self.personal_best_fitness[min_fitness_idx] < self.global_best_fitness:
                    self.global_best_fitness = self.personal_best_fitness[min_fitness_idx]
                    self.global_best_position = self.personal_best_positions[min_fitness_idx]

            self.provisional_minimum = self.provisional_min_computation()
            print("Provisional Minimum: ", self.provisional_minimum)
        print("Completed Optimization")
        return self.global_best_position, self.global_best_fitness, self.obj_comp_temp 

class HdFireflySimulatedAnnealingOptimizer:
    def __init__(self, ctsp_obj,  dimensions, range_max, pop_test=20, hdfa_iterations=100, gamma=1, alpha=.2): 
        self.ctsp_obj = ctsp_obj

        self.pop_test = pop_test 
        self.dimensions = dimensions
        self.hdfa_iterations = hdfa_iterations
        self.alpha = alpha
        self.gamma = gamma
        self.max_enums = math.factorial(self.ctsp_obj.num_cities)

        self.bounds = [range_max for _ in range(self.ctsp_obj.num_cities)]  
        self.max_depth = 3

        self.pop_positions = np.array([np.random.permutation(self.ctsp_obj.num_cities) for _ in range(self.pop_test)]) 
        self.pop_attractiveness = np.ones(self.pop_test)
        self.pop_fitness_true = np.zeros(self.pop_test)
        self.pop_alpha = np.ones(self.pop_test)

        self.firefly_positions_history = np.zeros((self.hdfa_iterations, self.pop_test, self.ctsp_obj.num_cities))
        self.minima_positions = [] 
        self.minima_fitness = []   
        self.fitness_history = []  
        self.obj_comp_temp = 0
        self.compute_fitness_true()
        self.bsp_tree = generate_bsp_tree(self.bounds, 0, self.max_depth)

    def compute_fitness_true(self):
        for idx in range(self.pop_test):
            #print("Initializing Firefly: ", idx)
            self.pop_fitness_true[idx] = self.ctsp_obj.total_distance(self.pop_positions[idx])
        self.obj_comp_temp = self.pop_test

    def initialize_positions(self, stage, low=None, high=None):
        if stage == 'initial':
            return low + (high - low) * np.random.rand(self.pop_test, self.dimensions)
        elif stage == 'finder_tracker':
            return self.finder_tracker_assignments()    
    
    def l2_norm(self, ff_idx_1, ff_idx_2):
        return np.sqrt(np.sum((self.pop_positions[ff_idx_1] - self.pop_positions[ff_idx_2])**2))

    def compute_attractiveness(self, idx1, idx2):
        norm = self.l2_norm(idx1, idx2)#**2
        self.pop_attractiveness[idx1] = self.pop_attractiveness[idx1] * np.exp(-self.gamma*norm)
        return self.pop_attractiveness[idx1]        

    def compute_positions(self):
        for idx1 in range(self.pop_test):
            for idx2 in range(self.pop_test):
                if self.pop_fitness_true[idx1] > self.pop_fitness_true[idx2]:
                    new_attr = self.compute_attractiveness(idx1, idx2)
                    distance = self.calculate_permutation_distance(self.pop_positions[idx1], self.pop_positions[idx2])

                    self.pop_positions[idx1] = self.update_position(self.pop_positions[idx1], self.pop_positions[idx2], new_attr, distance, idx1)

    def update_position(self, pos1, pos2, attractiveness, distance, idx1):
        perturbation = np.random.rand(len(pos1)) - 0.5
        perturbation *= self.pop_alpha[idx1]
        
        new_pos = pos1.copy()

        for i in range(len(pos1)):
            if np.random.rand() < attractiveness: 
                new_pos[i] = pos2[i]  

        for i in range(len(new_pos)):
            if np.random.rand() < 0.1:  
                j = np.random.randint(len(new_pos))
                new_pos[i], new_pos[j] = new_pos[j], new_pos[i]  
        new_pos = np.argsort(new_pos) 
        return new_pos
   

    def calculate_permutation_distance(self, perm1, perm2):
        return np.sum(np.array(perm1) != np.array(perm2))

    def sort_data(self):       
        fitness = np.argsort(self.pop_fitness_true)
        self.pop_positions = self.pop_positions[fitness]
        self.pop_attractiveness = self.pop_attractiveness[fitness]
        self.pop_fitness_true = self.pop_fitness_true[fitness]
        self.pop_alpha = self.pop_alpha[fitness]
        

    def finder_tracker_reassignments(self, tol=.5):
        print("Reassigning Finder Trackers")
        self.sort_data()
        for idx1 in range(self.pop_test):
            for idx2 in range(self.pop_test):            
                if self.l2_norm(idx1, idx2) < tol:
                    top_40 = int(self.pop_test * .4)
                    bottom_60 = int(self.pop_test * .6)
                    bottom_60_pos = self.pop_positions[top_40:]
                    bottom_60_fit = self.pop_fitness_true[top_40:]
                    bottom_60_alpha = self.pop_alpha[top_40:]

                    bottom_60_pos = np.array([np.argsort(np.random.permutation(self.ctsp_obj.num_cities)) for _ in range(bottom_60)])
                    bottom_60_fit = np.array([self.ctsp_obj.total_distance(bottom_60_pos[idx]) for idx in range(bottom_60)])
                    self.obj_comp_temp += bottom_60
                    bottom_60_alpha = np.ones(bottom_60)
                    top_40_pos = self.pop_positions[:top_40]
                    top_40_fit = self.pop_fitness_true[:top_40]
                    top_40_alpha = self.pop_alpha[:top_40]

                    self.pop_positions = np.concatenate((top_40_pos, bottom_60_pos))
                    self.pop_fitness_true = np.concatenate((top_40_fit, bottom_60_fit))
                    self.pop_alpha =  np.concatenate((top_40_alpha, bottom_60_alpha))
        
                    print("Reassignments Completed")                   
                    return    
        print("No Reassignments Needed")

    def optimize(self):
        print("Beggining Hd-Firefly-SA Optimization")
        last_alpha = float('inf')
        maturity_condition = False
        nonincreasing_alpha_counter = 0
        hdfa_ctr = 0
        min_reg_fitness = float('inf')
        new_fitness = float('inf')
        best_position = None
        best_fitness = float('inf')
        best_region = None
        while not maturity_condition:
            iteration_fitness = []        
            while hdfa_ctr < self.hdfa_iterations:
                print("Current HdFa Iteration: ", hdfa_ctr)

                self.compute_positions()
                self.compute_fitness_true()

                for idx in range(self.pop_test):

                    region = find_region(self.bsp_tree, self.pop_positions[idx])
                    
                    if region is None:
                        continue

                    neighbors = find_neighbors(region, 5)

                    is_best_region = find_min_fitness_region(region, neighbors, self.pop_positions[idx])

                    if is_best_region == region: 
                        new_fitness = self.ctsp_obj.total_distance(self.pop_positions[idx])
                        if new_fitness < region.fitness:
                            region.fitness = new_fitness
                        self.pop_alpha[idx] = np.abs(new_fitness - is_best_region.fitness)

                        region.split(new_fitness)
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_position = self.pop_positions[idx]
                            best_region = region
                                     
                    else:
                        self.pop_fitness_true[idx] = region.fitness
                     
                    iteration_fitness.append(self.pop_fitness_true[idx])                    
      
                alpha_avg = np.average(self.pop_alpha)    
                self.firefly_positions_history[hdfa_ctr] = self.pop_positions.copy()
                self.fitness_history.append(np.min(iteration_fitness))
                self.minima_positions.append(best_position)
                self.minima_fitness.append(best_fitness) 
                hdfa_ctr += 1
                print("Current Alpha Average: ", alpha_avg)

                print('HDFA Iteration: ', hdfa_ctr)
                print('Steps Without Increasing Alpha: ', nonincreasing_alpha_counter)

                if last_alpha >= alpha_avg:
                    nonincreasing_alpha_counter += 1
                else:
                    nonincreasing_alpha_counter = 0

                last_alpha = alpha_avg

                if nonincreasing_alpha_counter == 10 or alpha_avg == 0 or hdfa_ctr == 100 or math.isnan(nonincreasing_alpha_counter):
                    maturity_condition = True 
                    print("Early Convergence")                   
                    break                    
                self.finder_tracker_reassignments()

        print("HDFFA Optimization Complete")
        best_region = find_min_fitness_region_recursive(self.bsp_tree)
        
        best_pt_est = find_bounding_box_center(best_region.bounds)

        best_pt_est = np.argsort([round(val) for val in best_pt_est])

        print("Final Hd-FF Position Estimate: ", best_pt_est)
        print("Final Hd-FF Error: ", best_region.fitness)

        sa = SimulatedAnnealingOptimizer(self.ctsp_obj, best_pt_est, temperature=10, cooling_rate = .95)
        
        sa_min_pt, sa_min_fitness, obj_ct = sa.optimize()

        self.obj_comp_temp += obj_ct

        print("Final SA Min Position: ", sa_min_pt)
        print("Final SA Error: ", sa_min_fitness)
       
        return sa_min_pt, sa_min_fitness, self.obj_comp_temp 

