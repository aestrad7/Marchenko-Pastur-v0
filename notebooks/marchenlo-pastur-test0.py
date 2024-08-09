import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import chi2

# Generamos una matriz de datos aleatorios (n_samples x n_features)
n_samples = 1000
n_features = 300
X = np.random.randn(n_samples, n_features)
print(X[:5, :5])

# Realizamos el PCA
pca = PCA()
pca.fit(X)

# Extraemos los autovalores de la matriz de covarianza empírica
eigenvalues = pca.explained_variance_

# Calculamos el ratio p/n
c = n_features / n_samples

print(c)
# Calculamos los límites inferior y superior de la distribución de Marchenko-Pastur
lambda_minus = (1 - np.sqrt(c))**2
lambda_plus = (1 + np.sqrt(c))**2

print(f'Lower limit: {lambda_minus:.2f}')
print(f'Upper limit: {lambda_plus:.2f}')

# Identificamos autovalores significativos
significant_eigenvalues = eigenvalues[eigenvalues > lambda_plus]
print(f'Number of significant eigenvalues: {len(significant_eigenvalues)}')

# Graficamos la distribución de autovalores obtenidos del PCA
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=50, density=True, alpha=0.7, label='Eigenvalues (PCA)')

# Añadimos las líneas de los límites de Marchenko-Pastur
plt.axvline(lambda_minus, color='red', linestyle='--', label=f'$\lambda_-$: {lambda_minus:.2f}')
plt.axvline(lambda_plus, color='green', linestyle='--', label=f'$\lambda_+$: {lambda_plus:.2f}')

plt.title('Distribution of PCA Eigenvalues vs Marchenko-Pastur Limits')
plt.xlabel('Eigenvalue')
plt.ylabel('Density')
plt.legend()
plt.show()

