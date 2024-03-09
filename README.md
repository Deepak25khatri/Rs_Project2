# Book Recommendation System

## Overview

This project implements a collaborative filtering system for recommending books to users. It uses user-user, item-item, and matrix factorization techniques to provide personalized recommendations based on user preferences.

## Requirements

Ensure you have the following prerequisites to run the project:

- Python 3.x, numpy, pandas, scikit-learn, scipy, surprise, matplotlib.

## Setup

Install Jupyter and required libraries. Clone/download the repository.

## Usage

Navigate to the project folder, run jupyter notebook, and open the .ipynb file to explore the recommendation system.

## Work distribution
Nisarg Ganatra: I have worked on building the user-user and item-item based colaborative filtering system. In addition I have also developed an API for this.

Deepak Khatri:

Shubham Shah: My concentration was on the comparison part, specifically comparing the two systems: matrix factorization and user-user, item-item. To do this, I divided my training and testing data in an 80:20 ratio. Next, I determined the RMSE and MAE of those systems by predicting the ratings on the test data. The rmse and mae for matrix-factorization are relatively low, which makes it a more robust option for recommendation tasks. But for user-user and item-item, I had somewhat higher rmse and mae for various reasons such as Data sparsity, Limited test set , Cold-start problem, etc. 

Vikram: Mainly worked on data cleaning and EDA part and gained insights from that. 

### Front_End: https://github.com/Deepak25khatri/Recommendation_Frontend
### Backend : https://github.com/Deepak25khatri/Recommended_Backend

## References
https://towardsdatascience.com/recommender-systems-matrix-factorization-using-pytorch-bd52f46aa199
https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b
https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
