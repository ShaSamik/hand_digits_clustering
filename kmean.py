import random
import numpy as np


def kmeans_multiple_times(data, k, runs):

    best_sse = float('inf')
    best_clusters = None
    best_centers = None
    best_seed = None

    base_seed = 1007 # Found to be the best seed

    for i in range(runs):

        seed = base_seed + i
        random.seed(seed)
        np.random.seed(seed)

        print("Run", i + 1, "Seed:", seed)

        clusters, centers = kmeans(data, k)
        sse = compute_sse(data, centers, clusters)

        print("SSE for this run:", sse)

        if sse < best_sse:

            best_sse = sse
            best_clusters = clusters
            best_centers = centers
            best_seed = seed

    print("Best SSE:", best_sse)
    return best_clusters, best_centers, best_seed


# Sum of Squared Errors (SSE)
def compute_sse(data, centers, clusters):

    assigned_centers = centers[clusters]              
    squared_diff = (data - assigned_centers) ** 2
    total_sse = np.sum(squared_diff)
    return total_sse


# K-Means Algorithm
def kmeans(data, k, max_iters = 250):

    centers = initialize_centers(data, k)
    clusters = None

    for iteration in range(max_iters):

        new_clusters = assign_clusters(data, centers)
        new_centers = update_centers(data, new_clusters, k)

        if clusters is not None and np.array_equal(new_clusters, clusters):

            print("Converged after", iteration, "iterations")
            break

        clusters = new_clusters
        centers = new_centers

    return clusters, centers


# Pick starting cluster centers
def initialize_centers(data, k):

    indices = np.random.choice(data.shape[0], size = k, replace = False)
    centers = data[indices].copy()
    return centers


# Data Loader Function
def load_data(file_path):

    data = []

    with open(file_path, 'r') as file:

        for line in file:

            line = line.strip()

            if not line:

                continue

            values = np.array(line.split(','), dtype = float)

            values = clean_image(values) # threshold in 0-255 scale
            values = values / 255.0 # normalize to 0-1

            data.append(values)

    return np.array(data, dtype=float)


# Assign each image to nearest cluster center
def assign_clusters(data, centers):

    distances = np.sqrt(np.sum((data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    clusters = np.argmin(distances, axis=1)
    return clusters


# Update the centers
def update_centers(data, clusters, k):

    num_features = data.shape[1]
    new_centers = np.zeros((k, num_features), dtype=float)

    for cluster_id in range(k):

        cluster_points = data[clusters == cluster_id]

        if len(cluster_points) == 0:

            random_index = np.random.randint(0, data.shape[0])
            new_centers[cluster_id] = data[random_index]

        else:

            new_centers[cluster_id] = np.mean(cluster_points, axis=0)

    return new_centers


# Save output to file
def save_output(clusters, file_path):

    with open(file_path, 'w') as file:

        for cluster in clusters:

            file.write(str(int(cluster) + 1) + '\n')


# Clean up the image a little
def clean_image(image):

    return np.where(image <= 30, 0, image)


#################### EVALUATION CODE ###################
def evaluate_k_values(data):

    results = []

    for k in range(2, 21, 2):

        print("\nEvaluating K =", k)

        clusters, centers, best_seed = kmeans_multiple_times(data, k, runs=3)
        sse = compute_sse(data, centers, clusters)

        print("SSE for K =", k, ":", sse)
        print("Best Seed for K =", k, ":", best_seed)

        results.append((k, sse, best_seed))

    return results
#################### EVALUATION CODE ##################


# Mainnn
def main():
    file_path = "test.txt"
    output_file = "output.txt"

    k = 10
    runs = 1

    print("K-Means Clustering")

    # Load data
    data = load_data(file_path)

    print("\n[Data Summary]")
    print("Number of images       :", data.shape[0])
    print("Pixels per image       :", data.shape[1])
    print("Selected K             :", k)
    print("Number of runs         :", runs)

    print("\n[Clustering Started]")
    
    clusters, centers, best_seed = kmeans_multiple_times(data, k, runs)

    sse = compute_sse(data, centers, clusters)

    print("\n[Clustering Results]")
    print("Final SSE              :", sse)
    print("Seed                   :", best_seed)
    print("Cluster assignments    :", len(clusters))
    print("Cluster centers        :", len(centers))

    save_output(clusters, output_file)

    print("\n[Output Saved]")

    # Uncomment below to evaluate multiple K values

    # results = evaluate_k_values(data)
    # print("\n[K Evaluation Results]")

    # for k_value, sse_value, best_seed in results:

    #     print("K =", k_value, "SSE =", sse_value, "Best Seed =", best_seed)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    print("\nDone.")


# This executes the file
if __name__ == "__main__":
    main()