## Fixed Income Analytics: Bond Pricing and Risk Analysis

### Project Overview

This project is a detailed implementation of core fixed income analytical techniques, focusing on bond valuation and interest rate risk modeling. It emphasizes mathematically rigorous analysis of cash flows, duration, convexity, and stochastic simulation.

The primary objective is to price corporate and government bonds accurately and quantify their sensitivity to interest rate movements, culminating in a Monte Carlo-based Value at Risk (VaR) assessment for a sample portfolio.

The entire analysis, including the theoretical framework, functions, simulations, and visualizations, is contained within a single Jupyter Notebook: Bond_Risk_Analysis.ipynb.

### Repository Contents

File/Folder  
Description

Bond_Risk_Analysis.ipynb  
The primary Jupyter Notebook containing all financial functions, pricing routines, portfolio construction, Monte Carlo simulation, and visualizations.

README.md  
This file, providing project context and setup instructions.

### Key Methodologies and Techniques

The analysis employs core fixed income and statistical techniques, including:

- Discounted Cash Flow (DCF) Valuation: Implementation of the fundamental bond pricing formula using yield to maturity (YTM).
- Duration Measures: Analytical computation of Macaulay Duration and Modified Duration to quantify linear price sensitivity.
- Convexity: Calculation of bond convexity to measure curvature in the price-yield relationship.
- Portfolio Aggregation: Determining weighted-average duration and convexity for multi-bond portfolios.
- Sensitivity Analysis: Visualization of the cash flow schedule and the non-linear price-yield curve.
- Stochastic Modeling (Monte Carlo): Simulation of interest rate movements using duration and convexity approximations under assumed yield volatility.
- Value at Risk (VaR): Estimation of 95% and 99% confidence levels for portfolio losses based on simulated interest rate scenarios.

### Setup and Installation

#### Prerequisites

You need Python installed on your system. Using a virtual environment is strongly recommended for dependency management.

#### 2. Install Dependencies

The required data science libraries are listed below. If your system does not recognize pip directly, use python -m pip instead.

python -m pip install pandas numpy matplotlib seaborn scipy jupyter

#### 3. Launch the Notebook

After installation, start the Jupyter server:

jupyter notebook
