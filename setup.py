from setuptools import setup, find_packages

setup(name="pynamics",
      version="0.2.0",
      description="Python package intended to blend together control systems, machine learning, reinforcement learning, and optimisation algorithms.",
      author="Miguel Santos Loureiro",
      author_email="miguel.santos.loureiro@gmail.com",
      packages=find_packages(include=["pynamics", "pynamics.*"]),
      #packages=["pynamics", "pynamics.solvers", "pynamics.solvers.fixed_step"],
      install_requires=["numpy>=1.26.3", "pandas>=2.1.4"])