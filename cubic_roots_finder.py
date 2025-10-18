"""
Cubic Equation Roots Finder for Google Colab
============================================

This script provides multiple methods to find the roots of cubic equations
of the form: f(x) = ax³ + bx² + cx + d = 0

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
import warnings
warnings.filterwarnings('ignore')

class CubicRootsFinder:
    """
    A class to find roots of cubic equations using various numerical methods.
    """
    
    def __init__(self, a, b, c, d):
        """
        Initialize with coefficients of the cubic equation ax³ + bx² + cx + d = 0
        
        Parameters:
        -----------
        a, b, c, d : float
            Coefficients of the cubic equation
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    def f(self, x):
        """The cubic function f(x) = ax³ + bx² + cx + d"""
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d
    
    def f_prime(self, x):
        """First derivative f'(x) = 3ax² + 2bx + c"""
        return 3 * self.a * x**2 + 2 * self.b * x + self.c
    
    def f_double_prime(self, x):
        """Second derivative f''(x) = 6ax + 2b"""
        return 6 * self.a * x + 2 * self.b
    
    def analytical_roots(self):
        """
        Find roots using the analytical formula for cubic equations.
        Returns all three roots (real and complex).
        """
        a, b, c, d = self.a, self.b, self.c, self.d
        
        # Normalize coefficients
        if a == 0:
            raise ValueError("Coefficient 'a' cannot be zero for a cubic equation")
        
        # Convert to depressed cubic: t³ + pt + q = 0
        p = (3*a*c - b**2) / (3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
        
        # Discriminant
        delta = (q/2)**2 + (p/3)**3
        
        if delta > 0:  # One real root, two complex
            u = np.cbrt(-q/2 + np.sqrt(delta))
            v = np.cbrt(-q/2 - np.sqrt(delta))
            x1 = u + v - b/(3*a)
            x2 = -(u + v)/2 - b/(3*a) + 1j * np.sqrt(3) * (u - v)/2
            x3 = -(u + v)/2 - b/(3*a) - 1j * np.sqrt(3) * (u - v)/2
        elif delta == 0:  # Three real roots, at least two equal
            if q == 0:
                x1 = x2 = x3 = -b/(3*a)
            else:
                u = np.cbrt(-q/2)
                x1 = 2*u - b/(3*a)
                x2 = x3 = -u - b/(3*a)
        else:  # Three distinct real roots
            rho = np.sqrt(-p/3)
            theta = np.arccos(-q/(2*rho**3))
            x1 = 2*rho*np.cos(theta/3) - b/(3*a)
            x2 = 2*rho*np.cos((theta + 2*np.pi)/3) - b/(3*a)
            x3 = 2*rho*np.cos((theta + 4*np.pi)/3) - b/(3*a)
        
        return [x1, x2, x3]
    
    def numerical_roots(self, initial_guesses=None, method='hybrid'):
        """
        Find roots using numerical methods.
        
        Parameters:
        -----------
        initial_guesses : list, optional
            Initial guesses for the roots. If None, will generate automatically.
        method : str
            Method to use: 'newton', 'fsolve', or 'hybrid'
        
        Returns:
        --------
        list : Real roots found
        """
        if initial_guesses is None:
            # Generate initial guesses based on the function behavior
            x_range = np.linspace(-20, 20, 100)
            y_values = self.f(x_range)
            
            # Find sign changes to identify potential root locations
            sign_changes = []
            for i in range(len(y_values) - 1):
                if y_values[i] * y_values[i+1] < 0:
                    sign_changes.append((x_range[i] + x_range[i+1]) / 2)
            
            initial_guesses = sign_changes if sign_changes else [-10, 0, 10]
        
        roots = []
        
        for guess in initial_guesses:
            try:
                if method == 'newton':
                    root = newton(self.f, guess, fprime=self.f_prime, maxiter=1000, tol=1e-10)
                elif method == 'fsolve':
                    root = fsolve(self.f, guess)[0]
                else:  # hybrid
                    try:
                        root = newton(self.f, guess, fprime=self.f_prime, maxiter=1000, tol=1e-10)
                    except:
                        root = fsolve(self.f, guess)[0]
                
                # Check if root is real and not already found
                if np.isreal(root) and not any(abs(root - r) < 1e-6 for r in roots):
                    roots.append(float(root.real))
                    
            except:
                continue
        
        return sorted(roots)
    
    def plot_function_and_roots(self, x_range=(-10, 10), roots=None, figsize=(12, 8)):
        """
        Plot the cubic function and its roots.
        
        Parameters:
        -----------
        x_range : tuple
            Range of x values to plot
        roots : list, optional
            Roots to highlight on the plot
        figsize : tuple
            Figure size
        """
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = self.f(x)
        
        plt.figure(figsize=figsize)
        plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {self.a:.3f}x³ + {self.b:.3f}x² + {self.c:.3f}x + {self.d:.3f}')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        if roots:
            for i, root in enumerate(roots):
                plt.plot(root, 0, 'ro', markersize=8, label=f'Root {i+1}: x = {root:.6f}')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Cubic Function and its Roots')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def find_all_roots(self, plot=True, x_range=(-10, 10)):
        """
        Find all roots using both analytical and numerical methods.
        
        Parameters:
        -----------
        plot : bool
            Whether to plot the function and roots
        x_range : tuple
            Range for plotting
        
        Returns:
        --------
        dict : Results containing analytical and numerical roots
        """
        print("=" * 60)
        print("CUBIC EQUATION ROOTS FINDER")
        print("=" * 60)
        print(f"Equation: f(x) = {self.a:.6f}x³ + {self.b:.6f}x² + {self.c:.6f}x + {self.d:.6f}")
        print()
        
        # Analytical roots
        print("ANALYTICAL ROOTS (all roots including complex):")
        try:
            analytical = self.analytical_roots()
            for i, root in enumerate(analytical):
                if np.isreal(root):
                    print(f"  Root {i+1}: x = {root.real:.8f} (real)")
                else:
                    print(f"  Root {i+1}: x = {root:.8f} (complex)")
        except Exception as e:
            print(f"  Error in analytical method: {e}")
            analytical = []
        
        print()
        
        # Numerical roots (real only)
        print("NUMERICAL ROOTS (real roots only):")
        numerical = self.numerical_roots()
        if numerical:
            for i, root in enumerate(numerical):
                print(f"  Root {i+1}: x = {root:.8f}")
                print(f"  Verification: f({root:.8f}) = {self.f(root):.2e}")
        else:
            print("  No real roots found numerically")
        
        print()
        
        # Summary
        real_analytical = [r.real for r in analytical if np.isreal(r)]
        print("SUMMARY:")
        print(f"  Real roots found analytically: {len(real_analytical)}")
        print(f"  Real roots found numerically: {len(numerical)}")
        
        if plot:
            self.plot_function_and_roots(x_range=x_range, roots=numerical)
        
        return {
            'analytical': analytical,
            'numerical': numerical,
            'real_analytical': real_analytical
        }

def solve_cubic_equation(a, b, c, d, plot=True, x_range=(-10, 10)):
    """
    Convenience function to solve a cubic equation.
    
    Parameters:
    -----------
    a, b, c, d : float
        Coefficients of ax³ + bx² + cx + d = 0
    plot : bool
        Whether to plot the function and roots
    x_range : tuple
        Range for plotting
    
    Returns:
    --------
    dict : Results containing all roots
    """
    finder = CubicRootsFinder(a, b, c, d)
    return finder.find_all_roots(plot=plot, x_range=x_range)

# Example usage and test
if __name__ == "__main__":
    # Test with the provided equation: f(x) = 0.3x³ - 3.849069x² + 23.49069x - 44.179568
    print("Testing with the provided equation...")
    print("f(x) = 0.3x³ - 3.849069x² + 23.49069x - 44.179568")
    print()
    
    # Solve the equation
    results = solve_cubic_equation(0.3, -3.849069, 23.49069, -44.179568)
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS FOR GOOGLE COLAB:")
    print("="*60)
    print("1. Upload this script to your Colab notebook")
    print("2. Run the following code in a cell:")
    print()
    print("# Example 1: Solve the provided equation")
    print("results = solve_cubic_equation(0.3, -3.849069, 23.49069, -44.179568)")
    print()
    print("# Example 2: Solve a different cubic equation")
    print("results = solve_cubic_equation(1, -6, 11, -6)  # x³ - 6x² + 11x - 6")
    print()
    print("# Example 3: Use the class directly for more control")
    print("finder = CubicRootsFinder(1, -6, 11, -6)")
    print("results = finder.find_all_roots(plot=True)")
    print()
    print("# Example 4: Find only numerical roots without plotting")
    print("finder = CubicRootsFinder(1, -6, 11, -6)")
    print("roots = finder.numerical_roots()")
    print("print(f'Real roots: {roots}')")