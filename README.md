# Machine Learning 20: Practical work 5

> Simon Mirkovitch, Tiago Povoa Quinteiro

## Practical work 05 - Unsupervised Learning

> Use Self-Organized Maps (SOM) and K-Means as a means for clustering and visualization purposes using a collection of images. 

## Point 2. Clustering of wine data 

> Apply the K-means algorithm to the wine database. Use the 13 features of wine to find clusters in the data. a) Set K=3 and run K-means, 10 times each time, and b) given that you already know the type of wine for each observation, compute the average number (based on your 10 runs) of the number of observations that are correctly grouped together for each type of wine. Comment your results 

### a)

For the normalization, we used the following: (from sci-kit learn)

```python
from sklearn import preprocessing

preprocessing.normalize(data.data)
```

Parameters of the experiment: n_clusters=3, n_init=10, init=k-means++

We try to identify 3 classes.

There is 13 features by default. Only 7 on half

Below are the results of different tests. 10 runs each. We measure accuracy over all classes.

1. Run with no normalization and all the features

| 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0.16 | 0.35 | 0.16 | 0.18 | 0.18 | 0.18 | 0.16 | 0.16 | 0.16 | 0.70 |

2. Run with normalization  and all features

| 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0.69 | 0.25 | 0.69 | 0.20 | 0.69 | 0.25 | 0.25 | 0.25 | 0.20 | 0.69 |

3. Run with no normalization and half features

| 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0.18 | 0.47 | 0.16 | 0.16 | 0.18 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 |

4. Run with normalization and half features

| 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0.25 | 0.25 | 0.25 | 0.25 | 0.20 | 0.25 | 0.69 | 0.20 | 0.25 | 0.25 |

#### Summary

|                           | Mean |
| ------------------------- | ---- |
| Raw data, all features    | 0.23 |
| Normalized, all features  | 0.41 |
| Raw data, half features   | 0.19 |
| Normalized, half features | 0.28 |

The maximum score was obtained by the first experiment with 0.70 accuracy.

The best mean is with all features and normalization. Overall it gives more consistent results. 

Cutting by half the features clearly diminishes the accuracy. As this cut is arbitrary, it might remove very discriminant features. 

## Point 3. Clustering of images application 

> We will provide you with a database of color images, a set of three feature extraction methods and a SOM library. You may setup diverse experiments with the database, which contains a lot of classes. You should apply the three feature extraction methods and observe the results, and you have to modify the configuration and learning parameters of the SOM algorithm 

..