import numpy as np
import matplotlib.pyplot as plt

def least_squares_polynomial(x, y, degree):
    """
    Fits a polynomial of specified degree to data using the Least Squares Method.
    Equation: y = a0 + a1*x + a2*x^2 + ... + am*x^m
    """
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
      raise ValueError ("x and y must have same length")
    n = len(x)
    m = degree
    
    if n <= m:
        raise ValueError("Number of data points must be greater than the polynomial degree.")
        
    # Empty matrices for Normal equations
    A = np.zeros((m + 1, m + 1))
    B = np.zeros(m + 1)
    
    # Filling the matrices
    for row in range(m + 1):
        for col in range(m + 1):
            A[row, col] = np.sum(x**(row + col))
        B[row] = np.sum((x**row) * y)
    A[0,0] = n # the loop also does this with x^0 stating here just for clarity
    # Solve the system
    coeffs = np.linalg.solve(A, B)
    
    # Calculating the polynomial
    def eval_poly(x_vals, coefficients):
        y_vals = sum(coeff * x_vals**i for i, coeff in enumerate(coefficients))
        return y_vals

    # Plotting 
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label='Original Data', color='red')
    
    # High-resolution x array for smooth curve plotting
    x_plot = np.linspace(np.min(x), np.max(x), 100)
    y_plot = eval_poly(x_plot, coeffs)
    
    eq_terms = []
    for i, c in enumerate(coeffs):
      if i == 0:
        eq_terms.append(f"{c:.4f}")
      elif i == 1:
        eq_terms.append(f"{c:.4f}x")
      else:
        eq_terms.append(f"{c:.4f}x^{i}")
    eq_str = " + ".join(eq_terms).replace("+ -", "- ") # Removes -ve sign
    
    plt.plot(x_plot, y_plot, label=f"Fit (Degree {m})", color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Degree {m} Polynomial Least Squares Fit')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Statistics
    y_f = eval_poly(x, coeffs)
    y_mean = np.mean(y)
    
    SSY = np.sum((y - y_mean)**2)
    sr = np.sum((y - y_f)**2)

    sd = np.sqrt(SSY / (n - 1))
    sde = np.sqrt(sr / (n - (m + 1))) #adjusts DOF
    r2 = 1 - (sr / SSY)
    print(f"Fitted Equation: y = {eq_str}")
    print(f"Standard Deviation : {sd:.4f}")
    print(f"Standard Error of the Estimate : {sde:.4f}")
    print(f"Coefficient of Determination (R^2) : {r2:.4f}")
    
    return coeffs, r2

#example
x_data = [0, 1, 2, 3, 4, 5]
y_data = [2.1, 7.7, 13.6, 27.2, 40.9, 61.1]
least_squares_polynomial(x_data, y_data, degree=3)
