import requests
import json
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import entropy
import numpy as np


app = Flask(__name__)



drift_detected = False

@app.route('/',methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        threshold = float(request.form['threshold'])
        
    # Fetch data from the API endpoint
    url = "http://www.qts.iitkgp.ac.in/last/1/36000"
    response = requests.get(url)

    # Parse the response as JSON
    json_data = json.loads(response.text)

    # Extract y-values for plotting
    y = [d["Temperature"] for d in json_data]
    data = np.array(y)

    # Define a function to calculate the KL divergence between two probability distributions
    def kl_divergence(p, q):
        return entropy(p, q)

    # Define a function to estimate the initial data distribution using the reference set
    def estimate_initial_distribution(reference_set):
        # Estimate the mean and covariance of the data distribution using maximum likelihood
        mean = np.mean(reference_set, axis=0)
        cov = np.cov(reference_set.T)
        return mean, cov

    # Define a function to monitor the drift using the test set and the estimated data distribution
    def monitor_drift(test_set, reference_mean, reference_cov, threshold):
        # Estimate the current data distribution using the test set
        current_mean = np.mean(test_set, axis=0)
        current_cov = np.cov(test_set.T)

        # Calculate the KL divergence between the current and reference distributions
        kl_div = kl_divergence(np.hstack((current_mean, current_cov.flatten())), 
                            np.hstack((reference_mean, reference_cov.flatten())))
        
        # Compare the KL divergence to the threshold and return the result
        if kl_div > threshold:
            return True
        else:
            return False
        
    # Estimate the initial data distribution using the reference set
    reference_set = data[:24000] # Use the first 16000 samples as the reference set
    reference_mean, reference_cov =estimate_initial_distribution(reference_set)

    # Monitor the drift using the remaining data as the test set
    test_set = data[24000:]
    is_drift = monitor_drift(test_set, reference_mean, reference_cov,threshold)

    # Print the result
    if is_drift:
        print("Drift detected!")
        drift_detected=True

    else:
        print("No drift detected.")
        drift_detected=False
        

        # Create a plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data, color='#0072C6', linewidth=2)
        ax.set_xlabel('Data Point', fontsize=12)
        ax.set_ylabel('Temperature', fontsize=12)
        ax.set_title('Data Plot', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        # Encode the plot image as base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Render the template with the plot image
        return render_template('index.html', image=image_base64,drift_detected=drift_detected)
           
        
    

if __name__ == '__main__':
    app.run(debug=True)
