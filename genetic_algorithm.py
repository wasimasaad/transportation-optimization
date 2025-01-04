import numpy as np
import random
import math

class TransportOptimizer:
    def __init__(self, population_size=100,
            generations=100,
            mutation_rate=0.1,
            num_buses=3, 
            bus_speed=40, 
            stations=[(0, 0), (10, 10), (20, 20), (30, 30)]):
        """
        Initialize the optimization problem
        
        Args:
        - num_buses (int): Number of buses in the fleet
        - num_stations (int): Total number of stations
        - bus_speed (float): Average speed of buses (km/h)
        """
        self.num_buses = num_buses
        self.bus_speed = bus_speed
        self.stations = stations
        
        # Genetic algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Distances between stations (this would ideally be calculated based on actual coordinates)
        self.station_distances = self.generate_distances()
        self.num_stations = len(stations)

    def generate_distances(self):
        """
        Generate a distance matrix between stations
        """
        num_stations = len(self.stations)
        distances = np.zeros((num_stations, num_stations))
        for i in range(num_stations):
            for j in range(num_stations):
                if i != j:
                    # Calculate Euclidean distance between stations
                    x1, y1 = self.stations[i]
                    x2, y2 = self.stations[j]
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    distances[i, j] = distance
        
        # Ensure symmetry
        distances = (distances + distances.T) / 2
        return distances

    def generate_initial_population(self):
        """
        Generate initial population of timetables
        
        Each timetable is a 2D array where:
        - Rows represent buses
        - Values represent departure times at each station
        """
        population = []
        for _ in range(self.population_size):
            # Generate random departure times for each bus at each station
            timetable = np.random.uniform(0, 60, (self.num_buses, self.num_stations))
            
            # Ensure times are sorted for each bus (representing route order)
            timetable.sort(axis=1)
            
            population.append(timetable)
        return population
    
    def calculate_realistic_timetable(self, timetable):
        """
        Adjust timetable based on station distances and bus speed
        """
        for bus in range(self.num_buses):
            for station in range(1, self.num_stations):
                # Calculate minimum travel time from previous station
                distance = self.station_distances[station-1, station]
                min_travel_time = (distance / self.bus_speed) * 60  # Convert to minutes
                
                # Ensure arrival time respects minimum travel time
                min_arrival = timetable[bus, station-1] + min_travel_time
                timetable[bus, station] = max(timetable[bus, station], min_arrival)
        
        return timetable


    def calculate_distribution_score(self, timetable):
        """
        Calculate distribution score for buses
        Higher variance = better distribution = lower score (since we minimize)
        
        Args:
        - timetable (np.array): Departure times for buses at each station
        
        Returns:
        - float: Distribution score (lower is better)
        """
        # First make timetable realistic
        realistic_timetable = self.calculate_realistic_timetable(timetable.copy())
    
        distribution_score = 0
        
        # For each station, calculate distribution score
        for station in range(self.num_stations):
            # Get all bus arrival times at this station
            station_times = sorted(timetable[:, station])
            
            # Calculate time gaps between consecutive buses
            time_gaps = np.diff(station_times)
            
            if len(time_gaps) > 0:
                # Calculate coefficient of variation (standardized measure of dispersion)
                mean_gap = np.mean(time_gaps)
                std_gap = np.std(time_gaps)
                
                if mean_gap > 0:
                    # Lower coefficient of variation means more uniform distribution
                    cv = std_gap / mean_gap
                    distribution_score += cv
                
                # Penalize very small gaps (bunching)
                min_desired_gap = 5  # minimum 5 minutes between buses
                bunching_penalty = sum(1 for gap in time_gaps if gap < min_desired_gap)
                distribution_score += bunching_penalty * 2
        
        return distribution_score

    def crossover(self, parent1, parent2):
        """
        Create offspring by crossover
        
        Args:
        - parent1, parent2 (np.array): Parent timetables
        
        Returns:
        - Two offspring timetables
        """
        # Randomly choose crossover points
        crossover_point1 = random.randint(0, self.num_buses - 1)
        crossover_point2 = random.randint(0, self.num_stations - 1)
        
        # Create offspring
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        
        # Swap sections
        child1[crossover_point1:, crossover_point2:] = parent2[crossover_point1:, crossover_point2:]
        child2[crossover_point1:, crossover_point2:] = parent1[crossover_point1:, crossover_point2:]
        
        return child1, child2

    def mutate(self, timetable):
        """
        Mutate a timetable by slightly adjusting departure times
        
        Args:
        - timetable (np.array): Timetable to mutate
        
        Returns:
        - Mutated timetable
        """
        if random.random() < self.mutation_rate:
            # Choose a random bus and station
            bus = random.randint(0, self.num_buses - 1)
            station = random.randint(0, self.num_stations - 1)
            
            # Add small random adjustment
            timetable[bus, station] += random.uniform(-5, 5)
            
            # Ensure times remain sorted for each bus
            timetable[bus].sort()
        
        return timetable

    def optimize(self):
        """
        Run genetic algorithm to optimize timetables
        
        Returns:
        - Best timetable found
        """
        # Generate initial population
        population = self.generate_initial_population()
        
        for generation in range(self.generations):
            # Evaluate fitness (minimize waiting time)
            fitness_scores = [self.calculate_distribution_score(timetable) for timetable in population]
            
            # Sort population by fitness (lower is better)
            sorted_indices = np.argsort(fitness_scores)
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Keep best half of population
            population = population[:self.population_size // 2]
            
            # Generate new population through crossover and mutation
            while len(population) < self.population_size:
                # Select parents
                parent1, parent2 = random.sample(population, 2)
                
                # Perform crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate children
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                population.extend([child1, child2])
            
            # Truncate population to original size
            population = population[:self.population_size]
        
        # Return best timetable (lowest waiting time)
        best_timetable = min(population, key=self.calculate_distribution_score)
        return best_timetable

    def format_timetable(self, optimized_timetable):
        """
        Format the optimized timetable for output
        
        Args:
        - optimized_timetable (np.array): Optimized timetable
        
        Returns:
        - List of dictionaries with bus routes and times
        """
        bus_routes = []
        for bus_index, bus_times in enumerate(optimized_timetable):
            route = {
                'bus_number': bus_index + 1,
                'departure_times': [round(time, 2) for time in bus_times]
            }
            bus_routes.append(route)
        
        return bus_routes