# Master's Thesis Project: How to Get Rich by Fund of Funds Investment - An Optimization Method for Decision Making

## Overview

Optimal portfolios have historically been computed using standard deviation as a risk measure.However, extreme market events have become the rule rather than the exception. To capturetail risk, investors have started to look for alternative risk measures such as Value-at-Risk andConditional Value-at-Risk. This research analyzes the financial model referred to as Markowitz 2.0 and provides historical context and perspective to the model and makes a mathematicalformulation. Moreover, practical implementation is presented and an optimizer that capturesthe risk of non-extreme events is constructed, which meets the needs of more customized investment decisions, based on investment preferences. Optimal portfolios are generated and anefficient frontier is made. The results obtained are then compared with those obtained throughthe mean-variance optimization framework. As concluded from the data, the optimal portfoliowith the optimal weights generated performs better regarding expected portfolio return relativeto the risk level for the investment.

## Research Questions

1. To provide historical context and perspective to Markowitz 2.0 by reviewing the literature
on asset allocation models.
2. To make a mathematical formulation of Markowitz 2.0 based on relevant literature.
3. Implement Markowitz 2.0 in a programming language of choice and evaluate the model
in practice.


## Methodology

The data for the study is downloaded from Skandinaviska Enskilda Banken(SEB). SEB is the
third largest bank in Sweden and is a leading European financial group, with dealers dating
back to the year 1856. The data used is Fund data from 10 different funds from Sweden’s
biggest fund company; Swedbank Robur. The Swedbank Robur is composed of companies
from a variety of sectors and continents.

The model has been implemented in Python using the built-in package SciPy optimize which
implements portfolio optimization. The specific method used was SLSQP. The programming
code can be found in the Appendix A. SciPy optimize provides functions for maximizing (or
minimizing) objective functions, possibly subject to constraints. Constrained and non-linear
least-squares, linear programming, root finding, and curve fitting are among the non-linear
41
problem solvers the package offers. The following results in this section are optimizing an
objective function that maximize expected return. When we have a universe of assets or asset
classes with scenarios of asset returns for a certain period, a portfolio set with a set of equality
and non-equality restrictions, and a probability threshold where the loss is less than or equal to
the VaR, the CVaR optimization issue is fully described. The frequently employed probability
values are 0.99 and 0.95, which,

## Installation

git clone https://github.com/your-username/masterthesis.git
cd masterthesis
pip install -r requirements.txt
