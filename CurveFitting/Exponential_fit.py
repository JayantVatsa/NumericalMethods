import numpy as np
import matplotlib.pyplot as plt

def least_squares_exponential(x,y):
  """
    Fits an exponential curve (y = a*e^bx) to data using 
    the Least Squares Method.
    """
  
  x = np.array(x)
  y = np.array(y)
  
  if len(x) != len(y):
    raise ValueError("x and y must have same length")
  if np.any(y<=0):
    raise ValueError("y must be positive")
  
  l_y = np.log(y) # ln(y) since only y is transformed in the equation
  n = len(x) # length of the array
  
  # Summation
  sx = np.sum(x)
  sx2 = np.sum(x**2)
  sy = np.sum(l_y)
  sxy = np.sum(x*l_y)
  
  # Normal equations
  A = np.array([[n,sx],[sx,sx2]])
  B = np.array([[sy],[sxy]])
  
  # Solving the system
  X = np.linalg.solve(A,B)
  # Parameters
  a,b = X.flatten()
  a_exp = np.exp(a)
  
  # Fitted Linearised equation
  l_y_f = a + b*x
  
  # Fitted equation
  y_f = a_exp*np.exp(b*x)
  
  # Plotting
  plt.figure(figsize=(10, 5))
  plt.subplot(1,2,1) # Original data
  plt.scatter(x,y,label='Data points')
  x_plot = np.linspace(np.min(x), np.max(x), 100)
  y_plot = a_exp*np.exp(b*x_plot)
  plt.plot(x_plot,y_plot,label=f"Fitted curve: $y = {a_exp:.4f}e^{{{b:.4f}x}}$")
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.title('Fitted Exponential Law Curve')
  plt.grid()
  plt.legend()
  
  plt.subplot(1,2,2) # Linearised log plot
  plt.scatter(x,l_y,label=f"Linearised Data")
  l_y_plot = a + b * x_plot
  plt.plot(x_plot,l_y_plot,label=f'Fitted curve: ln(y) = {a:.4f} + {b:.4f}x ',color='red',linestyle='--')
  plt.xlabel('$x$')
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
  print(f"Standard Deviation : {sd:.4f}")
  print(f"Standard Error of the Estimate : {sde:.4f}")
  print(f"Coefficient of Determination : {r2:.4f}")
  return a_exp, b, r2   
#example
x = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
y = [2.5,1.5,0.8,0.7,0.35,0.30,0.20,0.10,0.10,0.03]
least_squares_exponential(x,y)
