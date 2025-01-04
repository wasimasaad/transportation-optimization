from flask import Flask, render_template, request, jsonify
from genetic_algorithm import TransportOptimizer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input parameters
        num_buses = int(request.form.get('num_buses', 3))
        bus_speed = float(request.form.get('bus_speed', 50))
        
        # Parse station coordinates
        # Get arrays of x and y coordinates from form
        station_x = request.form.getlist('station_x')
        station_y = request.form.getlist('station_y')
        
        # Convert to list of (x,y) tuples
        stations = list(zip(
            [float(x) for x in station_x],
            [float(y) for y in station_y]
        ))

        # get population size, generations and mutation rate
        population_size = int(request.form.get('population_size', 100))
        generations = int(request.form.get('num_generations', 100))
        mutation_rate = float(request.form.get('mutation_rate', 0.1))
        
        # Run optimization
        optimizer = TransportOptimizer(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            num_buses=num_buses, 
            bus_speed=bus_speed, 
            stations=stations
        )
        
        optimized_timetable = optimizer.optimize()
        bus_routes = optimizer.format_timetable(optimized_timetable)
    
        return render_template('index.html', bus_routes=bus_routes)

    return render_template('index.html')

if __name__ == '__main__':    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

    