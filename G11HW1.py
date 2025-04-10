import argparse
import math

from pyspark import RDD, SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans


def dist(u: tuple, C: tuple) -> float:
    min_i, min_d = 0, float('inf')
    for i, c in enumerate(C):
        d = math.dist(u, c)
        if d < min_d:
            min_i, min_d = i, d
    return min_i, min_d


def MRComputeStandardObjective(U: RDD, C: list) -> float:
    return (1 / U.count()) * U.map(lambda u: dist(u, C)[1] ** 2).sum()


def MRComputeFairObjective(U: RDD, C: list) -> float:
        return (U.map        ( lambda u:    (u[-1], (dist(u[:-1], C)[1] ** 2, 1)) ) # [(label, (distance, count)),...]
                 .reduceByKey( lambda x, y: (x[0] + y[0], x[1] + y[1])            ) # [(label, (sum_distance, sum_count)),...]
                 .map        ( lambda x:    (1 / x[1][1]) * x[1][0]               ) # [objective,...]       
                 .max())                                                            # max objective


def MRPrintStatistics(U: RDD, C: list) -> None:
    statistics = (U.map        ( lambda u: ((dist(u[:-1], C)[0], u[-1]), 1) ) # [((centroid, label), count),...]
                   .reduceByKey( lambda x, y: x + y                         ) # [((centroid, label), sum_count),...]
                   .map        ( lambda x: (x[0][0], (x[0][1], x[1]))       ) # [(centroid, (label, sum_count)),...]
                   .groupByKey()                                              # [(centroid, [(label, sum_count),...]),...]
                   .sortByKey()                                               
                   .collect())
    for centroid, label_counts in statistics:
        i      = centroid
        center = [f'{v:.6f}' for v in C[centroid]]
        NAi    = 0
        NBi    = 0
        for label, count in label_counts:
            if label == 'A': NAi = count
            if label == 'B': NBi = count
        print(f'i = {i}, center = ({",".join(center)}), NA{i} = {NAi}, NB{i} = {NBi}')


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
    delta = MRComputeStandardObjective(input_points.map(lambda x: x[:-1]), centroids.clusterCenters)
    print(f'Delta(U,C) = {delta:.6f}')
    phi = MRComputeFairObjective(input_points, centroids.clusterCenters)
    print(f'Phi(A,B,C) = {phi:.6f}')

    MRPrintStatistics(input_points, centroids.clusterCenters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the input file')
    parser.add_argument('L', type=int, help='Number of partitions')
    parser.add_argument('K', type=int, help='Number of centroids')
    parser.add_argument('M', type=int, help='Number of iterations')

    args = parser.parse_args()
    main(args.data_path, args.L, args.K, args.M)
