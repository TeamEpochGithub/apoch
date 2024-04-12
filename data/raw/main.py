if __name__ =="__main__":
    import numpy as np

    # Generating linearly separable data
    np.random.seed(42)  # For reproducibility

    # Parameters for the two classes
    n_samples = 100  # Number of samples per class
    mean1 = [2, 2]
    cov1 = [[1, 0], [0, 1]]  # Diagonal covariance matrix for class 1
    mean2 = [7, 7]
    cov2 = [[1, 0], [0, 1]]  # Diagonal covariance matrix for class 2

    # Generating samples
    data1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    labels1 = np.zeros((n_samples, 1))
    data2 = np.random.multivariate_normal(mean2, cov2, n_samples)
    labels2 = np.ones((n_samples, 1))

    # Concatenating the two sets of data to form a single dataset
    data = np.vstack((data1, data2))
    labels = np.vstack((labels1, labels2))
    dataset = np.hstack((data, labels))
    np.save('train_data/dummy.npy', dataset)




