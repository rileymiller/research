# Unsupervised Learning Investigation

- Clustering
   - for discrete problems
   - Used more than it should be because people assume an underlying domain has discrete classes in it. In reality data is continuous.
   - **K-means Clustering:** NP-hard problem, based on Euclidean distance. Algorithm Steps: a) Find the closest cluster center b) Recompute the cluster centroid. Solution isn't optimal.
   - **DBSCAN:** performs density-based clustering, follows the shape of neighborhood of points.

- Regressions
    - **Matrix Factorization:** model has more degrees of freedom than we have data. (Topic Modeling). From a factorization we can fill in missing values (Matrix Completion). Can be used with stochastic gradient descent. Issue with SGD is the local optima problem, where convergence happens in a local minima. Use MCMC (Markov-chain Monte-Carlo to reduce local optima issue, sometime method moves against the gradient. MCMC is the most accurate method for matrix factorization right now).

- Latent Semantic Indexing (LSI): identifies on words and phrases that occur frequently with each other

- Latent Derilicht Allocation

- Probabilistic latent semantic analysis

Can be used to extract topics from the text

Could be used to summarize a task

Text Categorization

https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df

https://medium.com/@fatmafatma/industrial-applications-of-topic-model-100e48a15ce4

