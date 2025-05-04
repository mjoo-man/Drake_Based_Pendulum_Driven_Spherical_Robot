import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from run_compare_steer_models import plot_drake_model
from ball_plant.data.plot_these_files import add_avg_data_to_plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)


if __name__=="__main__":
    
    ax_return, pipe_data, pend_data = add_avg_data_to_plot(ax)
    y_data = pipe_data["mean"]
    t_data = pipe_data["time"]
    q0 = [pipe_data["mean"][0], 0, pend_data["mean"][0], 0 ]

    tf = pipe_data["time"][-1] # final integrating time
    t = np.linspace(0, tf, 100)
    # Initial guess for parameters
    initial_params = [1.2e5, 0.75]

    # wrap the drake simulation to fit the problem
    def simulate_model(params):
        ax, data = plot_drake_model(None, q0, tf, config=("soft", "stiction"), new_proximity=params)
        return data["time"], data["pipe"]

    # Exponential weight on early data points
    def weight_function(t, tau=3.0):
        return np.exp(-t / tau)

    # Loss function: interpolate model to data timestamps, then compute weighted error
    def loss_function(params):
        t_model, y_model = simulate_model(params)
        interp_model = interp1d(t_model, y_model, bounds_error=False, fill_value="extrapolate")
        
        y_model_interp = interp_model(t_data)

        weights = weight_function(t_data)
        error = y_model_interp - y_data
        weighted_mse = np.mean(weights * error**2)
        return weighted_mse

    # Run optimization
    result = minimize(loss_function, initial_params, method='Nelder-Mead')

    # Output result
    print("Optimized parameters:", result.x)
    print("Final loss:", result.fun)
