import argparse

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans


def euclidean_distance(a, b):
    return sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** 0.5

def MRComputeStandardObjective(U, C):
    U = [u[:-1] for u in U]
    return (1 / len(U)) * sum([min([euclidean_distance(u, c) for c in C]) for u in U])


def MRComputeFairObjective(U, C):
    A = [u for u in U if u[-1] == 'A']
    B = [u for u in U if u[-1] == 'B']
    return max(MRComputeStandardObjective(A, C), MRComputeStandardObjective(B, C))


def MRPrintStatistics(U, C):
    counts = [{'A': 0, 'B': 0} for _ in range(len(C))]
    for u in U:
        closest_c = (0, float('inf'))
        for i, c in enumerate(C):
            dist = euclidean_distance(u[:-1], c)
            if dist < closest_c[1]:
                closest_c = (i, dist)
        counts[closest_c[0]][u[-1]] += 1

    for i, count in enumerate(counts):
        print(f'i = {i}, center = ({C[i][0]:.6f},{C[i][1]:.6f}), NA{i} = {count["A"]}, NB{i} = {count["B"]}')


def main(data_path, L, K, M):
    
    # Print command-line arguments
    print(f'Input file = {data_path}, L = {L}, K = {K}, M = {M}')

    # Setup Spark
    conf = SparkConf().setAppName('G11HW1')
    sc = SparkContext(conf=conf)
    
    # Subdivide the input file into L random partitions
    docs = sc.textFile(data_path).repartition(numPartitions=L).cache()
    input_points = docs.map(lambda x: [float(i) for i in x.split(',')[:-1]] + [x.split(',')[-1]]).cache()

    # Print N, NA, NB
    N = input_points.count()
    NA = input_points.filter(lambda x: x[-1] == 'A').count()
    NB = input_points.filter(lambda x: x[-1] == 'B').count()
    print(f'N = {N}, NA = {NA}, NB = {NB}')

    # Compute centroids without using labels
    centroids = KMeans.train(input_points.map(lambda x: x[:-1]), K, maxIterations=M)

    # Print standard and fair objectives
    delta = MRComputeStandardObjective(input_points.collect(), centroids.clusterCenters)
    print(f'Delta(U,C) = {delta:.6f}')
    phi = MRComputeFairObjective(input_points.collect(), centroids.clusterCenters)
    print(f'Phi(A,B,C) = {phi:.6f}')

    MRPrintStatistics(input_points.collect(), centroids.clusterCenters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the input file')
    parser.add_argument('L', type=int, help='Number of partitions')
    parser.add_argument('K', type=int, help='Number of centroids')
    parser.add_argument('M', type=int, help='Number of iterations')

    args = parser.parse_args()
    main(args.data_path, args.L, args.K, args.M)
