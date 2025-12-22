# Assignment 2

## Data Preparation
This folder contains the data-loading and preprocessing code for Assignment 2 of Airline Planning & Optimisation.
The goal is to load the provided Excel input files and compute all derived parameters in a clear, structured, and reproducible way, ready for use in the optimisation model.

### Structure
new_compute_parameters.py:
 - load airport, demand, hour coefficient, and fleet data from Excel
 - compute distances, yields, hourly demand, and operating costs
 - store everything in a single immutable ProblemData dataclass

new_main.py
 - Entry point showing how to build the data object and access parameters.

