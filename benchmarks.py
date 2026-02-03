import math

def ackley(d: dict[str, float]) -> float:
    """
    Ackley function. Many local minima.
    Global minimum is 0 at (0, 0).
    Recommended bounds: [-32.768, 32.768]
    """
    x, y = d.get('x', 0), d.get('y', 0)
    term1 = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2)))
    term2 = -math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
    return term1 + term2 + math.e + 20

def rosenbrock(d: dict[str, float]) -> float:
    """
    Rosenbrock function (The Banana Function).
    Global minimum is 0 at (1, 1). Hard to converge.
    Recommended bounds: [-5, 10]
    """
    x, y = d.get('x', 0), d.get('y', 0)
    a, b = 1, 100
    return (a - x)**2 + b * (y - x**2)**2

def rastrigin(d: dict[str, float]) -> float:
    """
    Rastrigin function. Highly multimodal (bumpy).
    Global minimum is 0 at (0, 0).
    Recommended bounds: [-5.12, 5.12]
    """
    x, y = d.get('x', 0), d.get('y', 0)
    A = 10
    return A * 2 + (x**2 - A * math.cos(2 * math.pi * x)) + (y**2 - A * math.cos(2 * math.pi * y))

def sphere(d: dict[str, float]) -> float:
    """
    Sphere function. Simple convex bowl.
    Global minimum is 0 at (0, 0).
    Recommended bounds: [-10, 10]
    """
    x, y = d.get('x', 0), d.get('y', 0)
    return x**2 + y**2

def beale(d: dict[str, float]) -> float:
    """
    Beale function. Sharp peaks and flat ridges.
    Global minimum is 0 at (3, 0.5).
    Recommended bounds: [-4.5, 4.5]
    """
    x, y = d.get('x', 0), d.get('y', 0)
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

# Registry of available functions
REGISTRY = {
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "sphere": sphere,
    "beale": beale
}

def get_function_info():
    return {
        name: func.__doc__.strip() for name, func in REGISTRY.items()
    }

def evaluate(name: str, params: dict) -> float:
    if name not in REGISTRY:
        raise ValueError(f"Function {name} not found.")
    return REGISTRY[name](params)