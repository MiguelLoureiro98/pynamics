from setuptools import setup, find_packages

setup(name="pynamics",
      version="0.1.0",
      description="Python package intended to blend together control systems, machine learning, reinforcement learning, and optimisation algorithms.",
      author="Miguel Santos Loureiro",
      author_email="miguel.santos.loureiro@gmail.com",
      packages=find_packages(["pynamics", "pynamics.*"]),
      install_requires=["numpy>=1.26.3", "pandas>=2.1.4"])