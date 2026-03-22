import numpy as np
import matplotlib.pyplot as plt

def least_squares_quadratic(x,y):
  """
    Fits a quadratic curve (y = a + bx + cx^2) to data using 
    the Least Squares Method.
    """
  x = np.array(x)
  y = np.array(y)
  if len(x) != len(y):
    raise ValueError("x and y must have same length")
  n = len(x)
  # Summation
  sx = np.sum(x)
  sx2 = np.sum(x**2)
  sx3 = np.sum(x**3)
  sx4 = np.sum(x**4)
  sy = np.sum(y)
  sxy = np.sum(x*y)
  sx2y = np.sum(x**2*y)
  # Normal equations
  A = np.array([[n,sx,sx2],[sx,sx2,sx3],[sx2,sx3,sx4]])
  B = np.array([[sy],[sxy],[sx2y]])
  # Solving the system
  X = np.linalg.solve(A,B)
  #Parameters
  a,b,c = X.flatten()
  # Fitted equation
  y_f = a + b*x + c*x**2
  # Plotting
  plt.figure(figsize=(10, 5))
  plt.scatter(x,y,label='Data points')
  # higher resolution x array for plotting the curve
  x_plot = np.linspace(np.min(x), np.max(x), 100)
  y_plot = a + b * x_plot + c * x_plot**2
  plt.plot(x_plot,y_plot,label=f"Fitted curve: y = {a:.4f} + {b:.4f}x + {c:.4f}x^2")
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.title('Fitted Quadratic Curve')
  plt.grid()
  plt.legend()
  plt.tight_layout()
  plt.show()
  # Statistics
  y_mean = np.mean(y)
  SSY = np.sum((y-y_mean)**2)
  sr = np.sum((y-y_f)**2)

  sd = np.sqrt(SSY/(n-1))
  sde = np.sqrt((sr)/(n-3))
  r2 = 1 - sr/SSY
  print(f"Standard Deviation : {sd:.4f}")
  print(f"Standard Error of the Estimate : {sde:.4f}")
  print(f"Coefficient of Determination : {r2:.4f}")
  return a, b, c, r2   
#example
x = [0,1,2,3,4,5]
y = [2.1,7.7,13.6,27.2,40.9,61.1]
least_squares_quadratic(x,y)
