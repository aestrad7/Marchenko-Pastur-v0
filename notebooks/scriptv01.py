import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataAnalyzer:
    """
    A class used to perform data analysis including loading data, standardizing features,
    visualizing correlation, performing PCA, and extracting eigenvalues.

    Attributes
    ----------
    data : DataFrame
        The loaded data.
    X : ndarray
        The data values.
    column_names : list
        The names of the columns in the data.
    X_scaled : ndarray
        The standardized data values.
    correlation_matrix : ndarray
        The correlation matrix of the standardized data.
    pca : PCA
        The PCA object used for performing PCA.
    X_pca : ndarray
        The data transformed by PCA.
    eigenvalues : ndarray
        The eigenvalues from the PCA.
    p_n_ratio : float
        The ratio of the number of features to the number of samples.
    lower_limit : float
        The lower limit for significant eigenvalues.
    upper_limit : float
        The upper limit for significant eigenvalues.
    significant_eigenvalues : ndarray
        The significant eigenvalues.
    """

    def __init__(self, data_file):
        """
        Initializes the DataAnalyzer with default values.
        """
        self.data_file = data_file
        self.data = None
        self.X = None
        self.column_names = None
        self.X_scaled = None
        self.correlation_matrix = None
        self.pca = None
        self.X_pca = None
        self.eigenvalues = None
        self.p_n_ratio = None
        self.lower_limit = None
        self.upper_limit = None
        self.significant_eigenvalues = None

    def load_data(self):
        """
        Loads data from a CSV file into a DataFrame and extracts the values and column names.
        """
        self.data = pd.read_csv(self.data_file)
        self.X = self.data.values
        self.column_names = self.data.columns.tolist()

    def standardize_features(self):
        """
        Standardizes the features of the data using StandardScaler.
        """
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def visualize_correlation(self):
        """
        Visualizes the correlation matrix of the standardized data using a heatmap.
        """
        self.correlation_matrix = np.corrcoef(self.X_scaled.T)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    xticklabels=self.column_names, yticklabels=self.column_names, 
                    cbar_kws={'shrink': .8}, linewidths=.5, linecolor='black')
        
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title('Correlogram', fontsize=16)
        plt.show()

    def perform_pca(self):
        """
        Performs Principal Component Analysis (PCA) on the standardized data.
        """
        self.pca = PCA()
        self.X_pca = self.pca.fit_transform(self.X_scaled)

    def scatter_plot_pca(self):
        """
        Creates a scatter plot of the first two principal components.
        """
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Scatter Plot')
        plt.show()

    def extract_eigenvalues(self):
        """
        Extracts the eigenvalues from the PCA.
        """
        self.eigenvalues = self.pca.explained_variance_

    def plot_eigenvalue_distribution(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.eigenvalues) + 1), self.eigenvalues, marker='o')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalue Distribution')
        plt.grid(True)
        plt.show()

    def calculate_p_n_ratio(self):
        self.p_n_ratio = len(self.column_names) / len(self.data)

    def calculate_marchenko_pastur_limits(self):
        self.lower_limit = (1 - np.sqrt(self.p_n_ratio))**2
        self.upper_limit = (1 + np.sqrt(self.p_n_ratio))**2

    def identify_significant_eigenvalues(self):
        self.significant_eigenvalues = self.eigenvalues[(self.eigenvalues > self.lower_limit) & (self.eigenvalues < self.upper_limit)]

    def print_significant_eigenvalues(self):
        print("Significant Eigenvalues:")
        for i, eigenvalue in enumerate(self.significant_eigenvalues):
            print(f"Eigenvalue {i+1}: {eigenvalue}")

    # def scatter_plot_pca_significant(self):
    #     """
    #     Creates a scatter plot of the first two principal components using only significant eigenvalues.
    #     """
    #     X_pca_significant = self.X_pca[:, :len(self.significant_eigenvalues)]
    #     plt.scatter(X_pca_significant[:, 0], X_pca_significant[:, 1])
    #     plt.xlabel('PC1')
    #     plt.ylabel('PC2')
    #     plt.title('PCA Scatter Plot (Significant Eigenvalues)')
    #     plt.show()

    #     # Calculate the correlation matrix of the projected data
    #     correlation_matrix_pca = np.corrcoef(X_pca_significant.T)
        
    #     plt.figure(figsize=(12, 10))
    #     sns.heatmap(correlation_matrix_pca, annot=True, fmt=".2f", cmap='coolwarm', 
    #                 xticklabels=self.column_names[:len(self.significant_eigenvalues)], 
    #                 yticklabels=self.column_names[:len(self.significant_eigenvalues)], 
    #                 cbar_kws={'shrink': .8}, linewidths=.5, linecolor='black')
        
    #     plt.xticks(rotation=90)
    #     plt.yticks(rotation=0)
    #     plt.title('Correlogram (Significant Eigenvalues)', fontsize=16)
    #     plt.show()

