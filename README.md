# UAV-AERO ENGINE

## Overview

UAV-AERO ENGINE is a modular engineering software project being developed
toward a SaaS platform for UAV aerodynamic analysis.

The long-term goal is to allow users to provide a 3D UAV model and receive
engineering analysis including geometry measurements, aerodynamic
performance, visualization, and generated reports.

Version 0.1.0 is the first internal engineering toolkit and establishes the
foundation for future development.

## Current Version

**v0.1.0 — Engineering Toolkit**

Status: Internal Prototype

## Current Capabilities

### Geometry

- Rectangle area
- Triangle area
- Circle area
- Trapezoid area
- Delta wing area
- Input validation
- Exception handling
- Automated tests

### Engineering Data

- CSV reading
- CSV writing
- Descriptive statistics
- Dataset filtering
- Processed-data export

### Visualization

- Altitude graph
- Battery graph
- Speed graph
- Wind graph
- PNG output

### Scientific Libraries

- NumPy
- Pandas
- Matplotlib
- SciPy
- trimesh

SciPy and trimesh are currently demonstrated through isolated experiments
and will become more important in future engineering modules.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- trimesh
- pytest
- Git
- GitHub

## Installation

Clone the repository and create a Python virtual environment.

```bash
python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

Install dependencies:

python -m pip install -r requirements.txt
Running Tests
pytest

Or:

python tests/run_tests.py
Running Scientific Experiments

SciPy:

python experiments/scipy_demo.py

trimesh:

python experiments/trimesh_demo.py

Project Structure

UAV-AERO-ENGINE/
│
├── geometry/
├── datasets/
├── visualization/
├── experiments/
├── reports/
├── tests/
├── docs/
│
├── main.py
├── requirements.txt
├── README.md
└── CHANGELOG.md

See the documentation in docs/ for more information.

Screenshots

Screenshots and visual demonstrations will be added as the project
develops.

Roadmap
Version	Focus
v0.1	Engineering Toolkit
v0.2	Engineering Mathematics
v0.3	Aerodynamics
v0.4	Geometry Engine
v0.5	Physics Engine
v0.6	Backend
v0.7	Frontend
v0.8	AI Engine
v0.9	Deployment
v1.0	Integrated SaaS Platform

License

This project is released under the MIT License.