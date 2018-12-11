# Logistic Regression on Titanic
A simple implementation of logistic regression on titanic data set using numpy.

this was done for a college assignment.

here we only used 3 features in titanic data: age, sex, pclass
the training is done for 100 epochs with different training batch sizes of 1, 5, 10 and 100 to compare the role of batch size hyperparameter on optimization.

```python
Summary of Training for 100 epochs:
Number of samples: 712
Batch Size: 1           Accuracy: 0.5197        Precision: 0.4427
Batch Size: 5           Accuracy: 0.7640        Precision: 0.6598
Batch Size: 10          Accuracy: 0.7107        Precision: 0.7591
Batch Size: 100         Accuracy: 0.6180        Precision: 0.5424

Summary of Test:
Number of samples: 179
Batch Size: 1           Accuracy: 0.4693        Precision: 0.4000
Batch Size: 5           Accuracy: 0.6983        Precision: 0.5579
Batch Size: 10          Accuracy: 0.7318        Precision: 0.7297
Batch Size: 100         Accuracy: 0.6592        Precision: 0.6111
```


confusion matrices are also plotted after training
![alt text](https://github.com/SadeghNK/Logistic-Regression-on-Titanic/raw/master/confusion%20matrices.PNG)
