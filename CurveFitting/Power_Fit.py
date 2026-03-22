import numpy as np
import matplotlib.pyplot as plt

def least_squares_power(x,y):
  """
    Fits a power law curve (y = a * x^b) to data using logarithmic transformation 
    and the Least Squares Method.
    """
  x = np.array(x)
  y = np.array(y)
  if len(x) != len(y):
    raise ValueError("x and y must have same length")
  if np.any(x <= 0) or np.any(y <= 0):
    raise ValueError("Power law fit requires x > 0 and y > 0")
  # Natural log transformation
  l_x = np.log(x)
  l_y = np.log(y)
  n = len(l_x)
  # Summation
  sx = np.sum(l_x)
  sxx = np.sum(l_x*l_x)
  sxy = np.sum(l_x*l_y)
  sy = np.sum(l_y)
  # Normal equations
  A = np.array([[n,sx],[sx,sxx]])
  B = np.array([[sy],[sxy]])
  # Solving the system
  X = np.linalg.solve(A,B)
  #Parameters
  a,b = X.flatten()
  # Fitted log equation
  l_y_f = a + b*l_x
  # Fitted power equation
  y_f = np.exp(a)*x**b
  # Plotting
  plt.figure(figsize=(10, 5))
  plt.subplot(1,2,1) # Original data
  plt.scatter(x,y,label='Data points')
  plt.plot(x,y_f,label=f"Fitted curve: y = {np.exp(a):.4f}x^({b:.4f})")
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.title('Fitted Power Law Curve')
  plt.grid()
  plt.legend()
  plt.subplot(1,2,2) # Linearised log plot
  plt.scatter(l_x,l_y,label=f"Linearised Data")
  plt.plot(l_x,l_y_f,label=f'Fitted curve: ln(y) = {a:.4f} + {b:.4f}ln(x) ',color='red',linestyle='--')
  plt.xlabel('$ln(x)$')
  plt.ylabel('$ln(y)$')
  plt.title('Linearised Fitted Curve')
  plt.grid()
  plt.legend(loc='lower right')
  plt.tight_layout()
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
  return np.exp(a), b, r2     
#example
x = [1,2,3,4,5]
y = [0.5,1.7,3.4,5.7,8.4]
least_squares_power(x,y)
