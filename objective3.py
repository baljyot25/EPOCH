
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

def distance_to_line(point, line):
    a, b = line
    projection = np.dot(np.eye(len(point)) - np.outer(a, a), point - b)
    return np.linalg.norm(projection)**2  

def cost_function(houses, line):
    return sum(distance_to_line(house, line) for house in houses)

california = fetch_california_housing()

x = california.data[:, 6]
y = california.data[:, 7]   
num=10#let



houses=np.asarray([x,y])


def find_maxs(line , x,y,k):
    distances = [(distance_to_line(np.asarray([ [x[i]], [y[i]] ]), line), x[i], y[i]) for i in range(len(x))]
    distances.sort(reverse=True,key=lambda x : x[0])
    return distances[:k]

def gradient_on_a_point(x_i,y_i, line):
    a, b = line
    m=a[1][0]
    c=b[1][0]
    grad_a = (m*x_i-y_i+c) * (m*y_i-m*c+x_i) * (1/(m**2+1))
    grad_b = (m*x_i-y_i+c) * (2/(m**2+1))
    return grad_a, grad_b


def determine_min_line(point , lines ):
    x_i,y_i=point
    point_array=np.asarray([[x_i],[y_i]])
    min_line=0
    mini_cost=distance_to_line(point_array,lines[0])
    for i in range(1, len(lines)):
        if (mini_cost<distance_to_line(point_array, lines[i])):
            min_line=i
            mini_cost=distance_to_line(point_array,  lines[i])
    return (min_line, mini_cost)

def optimize_line_b(max_iterations=5, learning_rate=0.00001,learning_rate_c=0.07, tolerance=1e-5):
    a = np.asanyarray([[1],[0]])*(1/np.sqrt(1))  
    b = np.asarray([[0], [0]])  
    line = (a, b)
    lines=[]
    for i in range(num):
        a = np.asanyarray([[1],[0]])*(1/np.sqrt(1))  
        b = np.asarray([[0], [0]])  
        lines.append((a,b))


    k=1
    for i in range(max_iterations):

        for j in range(len(x)):
            line_num , line_cost=determine_min_line((x[j],y[j]) ,lines )
            grad_a, grad_b = gradient_on_a_point(x[j],y[j], lines[line_num])
            new_a =np.asarray([[1],[a[1][0] - learning_rate * grad_a]])
            new_a /= np.linalg.norm(new_a)
            new_b =np.asarray([[0],[b[1][0]- learning_rate_c * grad_b]])

            new_line = (new_a, new_b)
            lines[line_num]=new_line
       
       
    return lines



lines = optimize_line_b(10)
for i in range(len(lines)):
    print(i+1 ,"th line ")
    a,b=lines[i]
    print(a, b, sep="\n")
    print()


