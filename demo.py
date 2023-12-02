# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# generate synthetic data
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 2))  # normal data
outlier_data = np.random.normal(loc=5, scale=1, size=(50, 2))    # outliers

# combine normal and outlier data
data = np.vstack([normal_data, outlier_data])

# standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# fit the Isolation Forest model
model = IsolationForest(contamination=0.05)  # 5% contamination
model.fit(data)

# predict anomalies
predictions = model.predict(data)

# plot the data points and highlight anomalies
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=predictions, cmap='viridis')
plt.title('Anomaly Detection Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Predicted Anomalies')
plt.show()
