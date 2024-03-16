
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

def distance_to_line(point, line):
    a, b = line
    projection = np.dot(np.eye(len(point)) - np.outer(a, a), point - b)
    return np.linalg.norm(projection)**2  # Squared distance

def cost_function(houses, line):
    return sum(distance_to_line(house, line) for house in houses)

# Example usage
# houses = np.array([[1, 2], [3, 4], [6 ,3]])  # Example coordinates of houses
# efficient_line, mc = optimize_line(cord)
# print("Minimum cost:", mc)
# print("Efficient Line:", efficient_line)

california = fetch_california_housing()

x = california.data[:, 6]
# x=x[:10000]
y = california.data[:, 7]   
# y=y[:10000]


houses=np.asarray([x,y])

def cost_function_b(x,y,line):
    return max(distance_to_line(np.asarray([[x[i]],[y[i]]]),line) for i in range(len(x)))

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

def optimize_line_b(max_iterations=5, learning_rate=0.00001,learning_rate_c=0.07, tolerance=1e-5):
    # Initialize line parameters
    a = np.asanyarray([[1],[0]])*(1/np.sqrt(1))  # Random initial direction vector
    # m /= np.linalg.norm(m)  # Normalize
    b = np.asarray([[0], [0]])  # Mean position of houses
    line = (a, b)

    cost = cost_function_b(x,y,line)

    # k =len(x)/6000
    k=1
    for i in range(max_iterations):
        arr=find_maxs(line,x,y,k)
        grad_a, grad_b = gradient_on_a_point(arr[0][1],arr[0][2], line)
        new_a =np.asarray([[1],[a[1][0] - learning_rate * grad_a]])
        new_a /= np.linalg.norm(new_a)
        new_b =np.asarray([[0],[b[1][0]- learning_rate_c * grad_b]])

        new_line = (new_a, new_b)
        new_cost = cost_function_b(x,y,new_line)
        # if abs(new_cost - cost) < tolerance:
        #     break
        a, b = new_a, new_b
        line = new_line
        cost = new_cost
        # mc = min(mc , cost)
    # print("cost is - ",mc)
    # print("cost is - ",cost)
    return line


for i in range(50, 51, 10):
    a, b = optimize_line_b(i)
    print(a, b, sep="\n")
    print("cost for ",i,"iterations " ,cost_function_b(x, y, (a, b)))


m = -0.13579492 # slope
c =-113.50287248

a = np.asanyarray([[1],[-0.933]])*(1/np.sqrt(1+(0.933)**2))  # Random initial direction vector
    # m /= np.linalg.norm(m)  # Normalize
b = np.asarray([[0], [-86.3204]])  # Mean position of houses
line = (a, b)


print("cost for ",1,"iterations " ,cost_function_b(x, y, (a, b)))

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, color='blue', label='Data Points')
plt.plot(x, m*x + c, color='red', label='Line: y = {}x + {}'.format(m, c))
plt.title('Relationship between Feature at Index 6 and Feature at Index 7')
plt.xlabel('Feature at Index 6')
plt.ylabel('Feature at Index 7')
plt.legend()
plt.grid(True)
plt.show()
