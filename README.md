### 1. Depth First Search (DFS)

```python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}
visited = set()

def dfs(visited, graph, node):
    if node not in visited:
        print(node, end=" ")
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

dfs(visited, graph, 'A')
```

---

### 2. Greedy Best-First Search

```python
import heapq

def greedy_best_first(graph, start, goal, heuristic):
    visited = set()
    pq = [(heuristic[start], start)]
    
    while pq:
        (h, node) = heapq.heappop(pq)
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            if node == goal:
                break
            for neighbor in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(pq, (heuristic[neighbor], neighbor))

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
heuristic = {'A': 5, 'B': 3, 'C': 2, 'D': 4, 'E': 1, 'F': 0}

greedy_best_first(graph, 'A', 'F', heuristic)
```
##### Easy
```python
import heapq as h
g={'A':['B','C'],'B':['D','E'],'C':['F'],'D':[],'E':['F'],'F':[]}
x={'A':5,'B':3,'C':2,'D':4,'E':1,'F':0}
v,p=set(),[(x['A'],'A')]
while p:
 _,n=h.heappop(p)
 if n in v:continue
 print(n,end=' ')
 if n=='F':break
 v.add(n)
 [h.heappush(p,(x[i],i))for i in g[n]]

```
---

### 3. Breadth First Search (BFS)

```python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}
visited = []
queue = []

def bfs(visited, graph, node):
    visited.append(node)
    queue.append(node)
    
    while queue:
        m = queue.pop(0)
        print(m, end=" ")
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

bfs(visited, graph, 'A')
```

---

### 4. Split Dataset into Train and Test Sets

```python
from sklearn.model_selection import train_test_split
import pandas as pd

data = {'X': [1,2,3,4,5,6,7,8], 'Y': [2,4,6,8,10,12,14,16]}
df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df[['X']], df['Y'], test_size=0.25, random_state=1
)

print(pd.concat([X_train, y_train], axis=1))
print(pd.concat([X_test, y_test], axis=1))
```

---

### 5. Decision Tree

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier()
clf.fit(X, y)

plt.figure(figsize=(10,6))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
```

---

### 6. Simple Linear Regression

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Prediction for x=6:", model.predict([[6]])[0])
```
