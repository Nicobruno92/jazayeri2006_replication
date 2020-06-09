### Preguntas a hacer
- TEngo que demostrar las funciones?\
por que el log?
- Solo tengo que resolver para random dot movement task? Tuning curve esta dada ya.
- Como generan los datos iniciales de las neuronas?
- Que es $\theta$ y $\theta_i$
- eje y. Figura 3

Curiosidad:
- Como hacen al figura 2? que usan?


```python
%load_ext lab_black
%matplotlib inline
```


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import scipy.io as sp
from scipy.stats import vonmises

# import statistics
# import pandas as pd
# from math import exp, sqrt
# import copy
# import random
```


```python
plt.style.use("ggplot")
fig_width = 8  # width in inches
fig_height = 6  # height in inches
fig_size = [fig_width, fig_height]
plt.rcParams["figure.figsize"] = fig_size
plt.rcParams["figure.autolayout"] = True

sns.set(style="white", context="notebook", palette="Set2", font_scale=1.5)
```

## circular standard deviation
 
$ SD = \sqrt{-2 \ln (R)} = \sqrt{\ln (1/R^2)} $
mus = [-2, -1, 0, 1, 2]

for i in mus:
    x = np.linspace(vonmises.ppf(0.01, kappa), vonmises.ppf(0.99, kappa), 100)
    plt.plot(x, vonmises.pdf(x, kappa, loc=i), lw=5, alpha=0.6, label="vonmises pdf")
# Optimal representationof sensory information by neural populations

$\log L(\theta) = \sum_{i = 1}^N \log L_i (\theta) = \sum_{i = 1}^N n_i \log fi(\theta)$

For Random dot motion

$\log L(\theta) = k \sum_{i = 1}^N n_i \cos \left(\theta - \theta_i \right)$

- $\theta$ = stimulus 
- $k$= determines the tuning binwidth

## Parameters
- $N = 720$ ==> number of neurons  
- $\kappa = 3$  ==> equivalent to a bandwidth of 90°
- $R_{min} = 10 imp/s$ ==> Minimum number of impulses use as threshold
- $R_{max} = 60 imp/s$ ==> Maximum response
- $\rho_{max} = 0.2$ ==> maximum correlation between 2 neurons, two neurons have the same preference
- $\delta = 0.1$ ==> ?
- $t = $
- $\theta_i = $
- $\theta$ ==> Stimulus presented


```python
N = 720
kappa = 3
R_min = 10
R_max = 60
rho_max = 0.2
delta = 0.1
t = 0.1
# theta_i = 
```
