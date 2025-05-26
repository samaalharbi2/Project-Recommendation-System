# 📚 Project: Recommendation System for IBM Community

This project focuses on building a **Recommendation System** using real interaction data from IBM's Watson Studio platform. The goal is to recommend articles to users based on their past behavior and similarities between articles or users.

----
## 💡 Project Overview

The recommendation system aims to answer the following questions:
- Which articles are most popular overall?
- What should we recommend to a new user?
- What should we recommend to a returning user based on their reading history?
- Can we find users that are most similar to a given user?

----
## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

----
### 📦 Dependencies
Make sure you have the following Python libraries installed:
```
pandas
numpy
sklearn
matplotlib
seaborn
scipy
```
----
### Installation

1- Clone the repository:

```
git clone url
cd Project-Recommendation-System
```
2- Open the notebook file in Jupyter Notebook or Jupyter Lab 

```
jupyter notebook Recommendations_with_IBM.ipynb
```
----
## Testing

Some tests are included in the notebook using assertion checks to verify correct implementation of key functions.

- 🔍 Breakdown of Tests
User-User Similarity Tests – Check if the most similar users are correctly identified.

Recommendations Tests – Ensure that recommended article IDs match expectations.

Cluster Assignments – Verify that articles are correctly mapped to clusters.

Submission Check – Export notebook and ensure outputs are correctly formatted.

----
## 📌 Project Instructions

The notebook includes the following sections:

- **Exploratory Data Analysis** – Understand the structure of the user-item interactions.
- **Rank-Based Recommendations** – Recommend articles based on popularity.
- **User-User Based Collaborative Filtering** – Recommend articles based on similar users.
- **Content-Based Recommendations** – Recommend articles based on clustering (e.g., KMeans).
- **Matrix Factorization (SVD)** – Use latent features to compute similarity.
- **Extras & Conclusion** – Build hybrid recommenders and polish results.
----
## Built With

* [Pandas](https://pypi.org/project/pandas/) - For data manipulation and analysis.
* [NumPy]([https://numpy.org/](https://pypi.org/project/numpy/)) - For numerical computing.
* [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) - For clustering, machine learning, dimensionality reduction, and evaluation metrics, including:
 * [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) - to measure similarity between items.
 * [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)- for clustering articles.
 * [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) - to convert text data into numerical features.
  * [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) - to build ML pipelines.
  * [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) - to normalize data.
  * [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) - for dimensionality reduction.
  * [model_evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)- for evaluating model performance.
* [Matplotlib](https://matplotlib.org/) - For visualizing data.
* [Jupyter Notebook](https://jupyter.org/) - For running and documenting Python code interactively.

Include all items used to build project.

----
## 🙏 Acknowledgment

Thanks to **Udacity** for providing this project as part of the **Data Scientist Nanodegree** program.

----
## License

[License](LICENSE.txt)
