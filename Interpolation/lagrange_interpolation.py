import numpy as np
import matplotlib.pyplot as plt

def lagrange_int(x,y,x0,n=None):
  """
    Evaluates the Lagrange interpolating polynomial at x0.
    """
  x = np.array(x,dtype=float)
  y = np.array(y,dtype=float)
  
  if len(x) != len(y):
    raise ValueError("x and y must have same length")

   if n is None: 
    n = len(x) - 1
     
  if n >= len(x):
    raise ValueError("n must be less than number of data points")

  px = 0.0 # lagrange's integrating polynomial

  for i in range(n+1): # for summing
    yi = y[i]
    for k in range(n+1): # for multiplying
      if k != i:
        if x[i] == x[k]:
          raise ValueError("x value can't be repeated")
        yi *= (x0 - x[k]) / (x[i] - x[k]) # cardinal function
    px += yi
  return px

# example
x = np.array([1,4,6])
y = np.array([0,1.386294,1.791760])

n1 = 1 # Linear
n2 = 2 # Quadratic
X = np.arange(x[0],x[len(x)-1]+0.25,0.25)
#for plotting
Y1 = np.empty(len(X))
Y2 = np.empty(len(X))
for j in range(len(X)):
  Y1[j] = lagrange_int(x,y,X[j],n1)
  Y2[j] = lagrange_int(x, y, X[j],n2)
plt.subplot(2,1,1)
plt.scatter(x,y,label='Data Points')
plt.plot(X,Y2,label='Interpolant Curve')
plt.title('Quadratic Interpolated Curve')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.legend()
plt.grid()
plt.tight_layout()
plt.subplot(2,1,2)
plt.scatter(x,y,label='Data Points')
plt.plot(X,Y1,label='Interpolant Curve')
plt.title('Linear Interpolated Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

