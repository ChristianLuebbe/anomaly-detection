Anomaly Detection
---

Talk at Applied Machine Learning Days 2020 workshop - [Workshop slides](https://docs.google.com/presentation/d/1Jg9rO_3dXwKzJyDOr2ley8Is5oWKE6D_aJJlJrpw0mw)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChristianLuebbe/anomaly-detection)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ChristianLuebbe/anomaly-detection/master)

**Dataset**

The data is based on the [KDD-CUP 1999 challenge](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) on network intrusion detection. A description of the original task can be found [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html). The data provided for this workshop has been adapted from the [NSL-KDD version](https://www.kaggle.com/hassan06/nslkdd).

**Anomaly detection**

Anomaly detection can be treated as a supervised classification task. However this approach struggles when the portion of anomalies (here network attacks) is small. Instead we showcase an approach using [Isolation Forests](https://www.youtube.com/watch?v=RyFQXQf4w4w). 

The user can select the size of training dataset and vary its contamination rate, including a dataset without any anomalies. The model is then trained on this dataset and used to predict anomalies on a separate test set and evaluate the performance.