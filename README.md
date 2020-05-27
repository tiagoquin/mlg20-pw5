# Machine Learning 20: Practical work 5

> Simon Mirkovitch, Tiago Povoa Quinteiro

## Practical work 05 - Unsupervised Learning

> Use Self-Organized Maps (SOM) and K-Means as a means for clustering and visualization purposes using a collection of images. 

## Point 1.  Explore the use of Self-Organizing Maps and K-Means  

The answers to the questions are in the notebook SOM_part1.ipynb

## Point 2. Clustering of wine data 

> Apply the K-means algorithm to the wine database. Use the 13 features of wine to find clusters in the data. a) Set K=3 and run K-means, 10 times each time, and b) given that you already know the type of wine for each observation, compute the average number (based on your 10 runs) of the number of observations that are correctly grouped together for each type of wine. Comment your results 

### a)

For the normalization, we used the following: (from sci-kit learn)

```python
from sklearn import preprocessing

preprocessing.normalize(data.data)
```

Parameters of the experiment: **n_clusters=3, n_init=10, init=k-means++**

We try to identify **3** classes.

There is **13** features by default. Only **7** on half

Accuracy was measured as: `[correctly classified] / [all elements]` (nothing fancy here)

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

### Data

We have 1000 images at our disposal. Each category is made of 100 pictures.

* 0-99: Pictures from African culture 
* 100-199: Beaches
* 200-299: Monuments
* 300-399: Buses
* 400-499: Dinosaurs
* 500-599: Elephants
* 600-699: Flowers
* 700-799: Horses
* 800-899: Mountains
* 900-999: Food

They all measure the same size **384x256** and roughly the same weight around **30~50 kB**.

### Extracting methods

We have three methods:

```python
    if method == 1:
        histograms = extractor.extract_histogram()
    elif method == 2:
        histograms = extractor.extract_hue_histogram()
    elif method == 3:
        histograms = extractor.extract_color_histogram()
```

> More information about these methods WangImageUtilities.py. 

The three histograms are:

1. Gray intensity - **10** features

   Extracts features from the gray shades in the picture. They pass first in a gray scale filter.

2. Hue - **10** features

   Extracts features from the hue specter from 0 to 360. A hue function represents the more intense areas by color groups. 

3. Color intensity - **30** features

   Extract features from the image color. Only method having 30 features (instead of 10)

### Number of iterations

I picked **10'000** since it doesn't take that much time on my machine.

As mentioned in the course, k-means isn't subject to over-fitting. It is not comparable to what happens in supervised learning. 

In my observations, changing the number of iterations didn't change by much the results (almost no difference). The only observation was that the dots in the Kohonen map were, in general, slightly more specific (presence of lighter AND darker dots around the map, in opposition of a more blend medium gray).

### Experiments

#### Procedure

For the three experiments, I tried the three methods. Generating the Kohonen map and the picture map.

#### Experiment 1: Elephants and flowers

##### Parameters

For the first experiment we are trying to solve the classification of two classes: flowers and elephants.

Parameters:

* Kohonen map 10x10
* Number of images: **200** (range 500-600, 600-700)
* Iterations: **10'000**

##### Gray intensity

Below is the Kohonen plot of the u-matrix.

<img src="./_img/i1-1.png" alt="koho1-1" style="zoom:67%;" />

Let's discuss this first experiment.

The first observation we can make is that the blue prototypes above correspond to the blank spots below. That's because they represent the unused prototypes. 

We can see a clear horizontal separation between the elephants in higher part, and the flowers in the lower part. 

In the middle of the picture, we have darker circles. Here we have a lot of difference. One is particularly interesting  two points above the lower blank spot (blue prototype). In this area we have a lot of difference. 

Around the middle of the Kohonen map, we have some mixed up things. Some elephants get lower the horizontal middle. And if we see the generated `.html` file, we can observe some mistakes. There is a few flowers and elephants who are classified together in a single prototype. 

Despite some mistakes, the overall classification looks good.

<img src="./_img/e1-1.png" alt="koho1-1" style="zoom:67%;" />

##### Hue method

<img src="./_img/i1-2.png" alt="koho1-2" style="zoom:67%;" />

This time, we can see the both maps for the hue method. 

Clearly the method is less reliable to classify these two classes.

Everything is spread all over the place. Same for the prototypes with both elephants and flowers.

<img src="./_img/e1-2.png" alt="koho1-2" style="zoom:67%;" />

##### Color intensity

<img src="./_img/i1-3.png" alt="koho1-3" style="zoom:67%;" />

The last method uses color to extract 30 features. 

It cuts out the classes by diagonal. There is only a few mistakes in the `.html` in the middle-left corner.

<img src="./_img/e1-3.png" alt="koho1-3" style="zoom:67%;" />

##### Conclusion

In this first experiment, the hue method is the worst choice.

The performance between gray and color methods is debatable. The cut in gray intensity is nicer, but the color one seems to have less mistakes (prototypes with both classes). 

#### Experiment 2: Beaches, Monuments and Mountains

For this experiment, we wanted to challenge the system with a difficult task. All these categories look more alike. Plus having three of them might turn this even harder.

##### Parameters

Three classes: beaches, monuments and mountains.

Parameters:

* Kohonen map 10x10
* Number of images: **300** (range 100-200, 200-300, 800-900)
* Iterations: **10'000**

##### Gray intensity

<img src="./_img/i2-1.png" alt="koho2-1" style="zoom:67%;" />

Here, no prototype is unused. 

The classification is a disaster. No clear cut between any class. We see a bit more beaches in the right, and a bit more monuments in the left. As for the mountains, they are spread everywhere. You can see them in the `.html` file.

<img src="./_img/e2-1.png" alt="koho2-1" style="zoom:67%;" />

##### Hue method

<img src="./_img/i2-2.png" alt="koho2-2" style="zoom:67%;" />

Same as in the previous section. No prototype is unused. Everything seems random.

<img src="./_img/e2-2.png" alt="koho2-2" style="zoom:67%;" />

##### Color method

<img src="./_img/i2-3.png" alt="koho2-3" style="zoom:67%;" />

Using this method, we got three unused prototypes. 

The upper part is composed of beaches and the lower part of monuments.

At first glance, we can at least see some horizontal distinction, despite having still lot of mistakes inside the prototypes. And the mountains are still not very well classified.

<img src="./_img/e2-3.png" alt="koho2-3" style="zoom:67%;" />

##### Conclusion

This experiment was built on purpose to test the limits of this system. So, as one could expect, it does poorly. 

No matter the method, we don't get a good classification. However we have some hope with the last method.

#### Experiment 3: Buses, horses and food

In the previous sections, we explored both an easy and a harder problem. Time for a middle ground. We expect this problem to be doable but still difficult.

##### Parameters

Three classes again: buses, horses and food.

Parameters:

* Kohonen map 10x10
* Number of images: **300** (range 300-400, 700-800, 900-1000)
* Iterations: **10'000**

##### Gray intensity

<img src="./_img/i3-1.png" alt="koho3-1" style="zoom:67%;" />

All the prototypes are used. We have all the horses on the left with a vertical cut. 

By looking at the `.html`, we can find where all the food pictures land. They are placed mostly in the middle.

We have a good right corner for the buses. 

<img src="./_img/e3-1.png" alt="koho3-1" style="zoom:67%;" />

##### Hue intensity

<img src="./_img/i3-2.png" alt="koho3-2" style="zoom:67%;" />

This classification leaves three empty prototypes.

The down side is cut and well defined for the horses.

The left side is mostly correct with the buses.

The middle and right side is mixed with the food pictures and the rest.

<img src="./_img/e3-2.png" alt="koho3-2" style="zoom:67%;" />

##### Color intensity

<img src="./_img/i3-3.png" alt="koho3-3" style="zoom:67%;" />

Same as in the grey intensity method, the horses are all classified on the left. We got three unused prototypes.

A lot of food is classified on the bottom, and by looking at the generated file, also in the right top corner. 

The buses are more spread but mostly in the middle.

The blue buses are correctly classified. The prototypes who found red and yellow buses are less accurate.

<img src="./_img/e3-3.png" alt="koho3-3" style="zoom:67%;" />

### Conclusion

In the experiment with the flowers and elephants:

The gray intensity gives a nice horizontal cut. The two others have more unused prototypes but a bit more spread. On the rate of mistakes, they looked roughly equivalent.

The second experiment with beaches, monuments and mountains was way harder. Only the color intensity method gave a hint of a somehow usable result. 

In the last experiment, we could observe that food was harder to classify. Since the pictures vary a lot in shape and color. But the horses and the buses were always clearly separated. The worst case scenario would be a brown bus with a green landscape behind.

Overall, the gray intensity method made less unused prototypes. All along this report I mentioned it, because having more holes means we have more pictures in the same spots, hence a more efficient system. The one that convinced me more was the color intensity method.

In conclusion, we can tell that k-means is very efficient in easy distinguishable scenarios, but struggles a lot in ambiguous ones. 