# Model Investigation
Thoughts on potential ML model architecture and algorithms.

- Baseline comparison with shallow models. Key in on logistic regression which is commonly used in recommending systems.

- Deep learning as a standalone approach, faulty for overgenerralizing.

- Hybrid Approach

- Something else to think about is the potential of unsupervised learning to act as an agnostic approach to all micro job platforms.

### Deep Learning with Python
Introduction to Keras and TensorFlow (TF). Either going to use TF or PyTorch to build model.

In order to do deep learning based recommendations I'm going to need a ton of data.

Need to do some type of learning based off user data (based on the user profile and other users, what task is the user most likely to select).

### Goals of the Model
Need to consider the primary goals of the machine learning model. This needs to be defined to drive the research goals of this development. Should we be trying to help users make the most money? Do we want user's to acquire new skills? High-level we spoke about trying to match the best workers for the task which we hypothesize would result in a higher quality of the task.

### 11/17 Thoughts
Going to need to take a content based approach due to the nature of the problem that crowdsourcing platforms inherently present where collaborative filtering won't be a viable option due to the frequent turnover of tasks and the customization aspect of having an agnostic model for general crowdsourcing platforms.

### Math Links
https://medium.com/data-science-bootcamp/understand-dot-products-matrix-multiplications-usage-in-deep-learning-in-minutes-beginner-95edf2e66155

https://en.wikipedia.org/wiki/Recurrent_neural_network

https://towardsdatascience.com/


### 11/20 Recommendation Algorithms
algorithms for recommender systems
https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed

pt. 2
https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-2-deep-recommendation-sequence-prediction-automl-f134bc79d66b

### Text Classification
https://towardsdatascience.com/automated-text-classification-using-machine-learning-3df4f4f9570b


### Skills extraction
Job Skills extraction with LSTM and Word Embeddings
 A graph-based approach to skill extraction from text.

**Entity Relations:**https://www.nltk.org/book/ch07.html
