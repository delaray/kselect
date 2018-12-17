# kselect

## Overview

This module implements an algorithm for determiming a near optimal K for use with the K-Means clustering algorithm. It applies the welknow Elbow Method along with the Bayesian Information Criteria (BIC)

## Installation

This module requires numpy, pandas, scipy and sklearn.

## Demo

## Experiments

Because the BIC is based on cluster density, there is a some variation in the value selected for K depending on the total number of data points. Several expereminents were performed using artificial clusters arranged in square grid with data points randomly generated in each square of the grid. A sufficient margin was imposed between the squares so the clusters are easily visible to the human eye.


| correct_k |  data_size |  margin |  predicted_k |  error |
| --------: | ----------:| -------:| ------------:| ------:|
|         4 |       1000 |      50 |           12 |     -8 |
|         4 |       2000 |      50 |           14 |    -10 |
|         4 |       3000 |      50 |           27 |    -23 |
|         4 |       4000 |      50 |           25 |    -21 |
|         9 |       1000 |      50 |           13 |     -4 |
|         9 |       2000 |      50 |           27 |    -18 |
|         9 |       3000 |      50 |           27 |    -18 |
|         9 |       4000 |      50 |           25 |    -16 |
|        16 |       1000 |      50 |           12 |      4 |
|        16 |       2000 |      50 |           13 |      3 |
|        16 |       3000 |      50 |           24 |     -8 |
|        16 |       4000 |      50 |           25 |     -9 |
|        25 |       1000 |      50 |           27 |     -2 |
|        25 |       2000 |      50 |           28 |     -3 |
|        25 |       3000 |      50 |           28 |     -3 |
|        25 |       4000 |      50 |           27 |     -2 |
|        36 |       1000 |      50 |           26 |     10 |
|        36 |       2000 |      50 |           26 |     10 |
|        36 |       3000 |      50 |           26 |     10 |
|        36 |       4000 |      50 |           26 |     10 |


## Examples

![alt text](https://github.com/delaray/kselect/img/kselect-25.png)