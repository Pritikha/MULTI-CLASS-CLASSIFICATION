# MULTI-CLASS-CLASSIFICATION
## Aim:
To write a python program to implement the multi class classification algorithm .

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner / Google Colab

## Related Theoritical Concept:
Neural networks reflect the behaviour of the human brain. They allow programs to recognise patterns and solve common problems in machine learning. This is another option to either perform classification instead of logistics regression.

Classification is categorizing objects into groups. There are different types of classifications one of which is multiclass classification.In multi-class classification, the neural network has the same number of output nodes as the number of classes. Each output node belongs to some class and outputs a score for that class.

Class is a category for example Predicting animal class from an animal image is an example of multi-class classification, where each animal can belong to only one category.

Scores from the last layer are passed through a SoftMax layer. The SoftMax layer converts the score into probability values. At last, data is classified into a corresponding class, that has the highest probability value

![164497964-1bee8c4e-0b2a-4b9e-af3a-8b1f538dd12e](https://user-images.githubusercontent.com/78891075/168518108-78c97655-9be4-48c1-839e-7122203b52c6.png)




## Algorithm
1.Import necessary libraries from packages.
2.Assign x,y values from the given dataset by sklearn.
3.Count the number of key-value pairs in the datasets using counter libraries.
4.Plot the x and y values in the chart using mathplotlib.pyplot.
5.Label the values of x and y axis and add title to the graph.
6.Save the file and execute the program.


## Program:

Program to implement the multi class classifier.

Developed by: Pritikha A P

RegisterNumber: 212219040114

```python3

from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot
X, y = make_blobs(n_samples=1000, centers=3, random_state=1)
print(X.shape, y.shape)
counter = Counter(y)
print(counter)
for i in range(10):
    print(X[i], y[i])
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

```

## Output:
![ss3](https://user-images.githubusercontent.com/78891075/168518537-f5d8bf27-ac08-4576-a900-cc7367012b9f.png)




## Result:
Thus the python program to implement the multi class classification was implemented successfully.
