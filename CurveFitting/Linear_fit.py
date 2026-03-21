import numpy as np
import matplotlib.pyplot as plt

def least_squares_linear(x,y):
  """
    Fits a straight line (y = intercept + slope * x) to data using the Least Squares Method.
    """
  x = np.array(x)
  y = np.array(y) 
  n = len(x)
  # Summations
  sx,sy = np.sum(x),np.sum(y)
  sxx,sxy = np.sum(x*x),np.sum(x*y)
  # Normal Equations
  A = np.array(([[n,sx],[sx,sxx]]))
  B =  np.array([[sy],[sxy]]) 
  # Solve the system of equations
  X = np.linalg.solve(A,B) 
  intercept,slope = X.flatten()
  # Fitted equation
  y_f = intercept + slope*x 
  print(f"The fitted line's equation is y = {intercept:.4f} + {slope:.4f}x")
  # Plotting
  plt.scatter(x,y,color='red',label='Original data points')
  plt.plot(x,y_f,color='blue',linestyle='--',label=f"Fitted line, y ={intercept:.4f} + {slope:.4f}x")
  plt.grid()
  plt.xlabel('x')
  plt.ylabel('f(x)')
  plt.title('Least Squares Method')
  plt.legend()
  plt.show()
  # Statistics
  y_mean = np.mean(y)
  SSY = np.sum((y-y_mean)**2)
  sr = np.sum((y-y_f)**2)
  
  sd = np.sqrt(SSY/(n-1))
  sde = np.sqrt((sr)/(n-2))
  r2 = 1 - sr/SSY
  r = np.sqrt(r2)
  print(f"Standard Deviation : {sd:.4f}")
  print(f"Standard Error of the Estimate : {sde:.4f}")
  print(f"Coefficient of Determination : {r2:.4f}")
  return intercept, slope, r2

#Example
x = [1,2,3,4,5,6,7]
y = [0.5,2.5,2.0,4.0,3.5,6.0,5.5]
least_squares_linear(x,y)
