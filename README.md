# Unsupervised Sentiment Analysis of IMDB Reviews using RoBERTa & K-Means



This project explores an unsupervised approach to sentiment analysis on the IMDB movie review dataset. It leverages embeddings from the powerful **RoBERTa** language model, reduces their dimensionality using an **Autoencoder**, and then performs clustering with **K-Means** to separate positive and negative reviews without using the original labels for training the final classifier.

---

## 📖 Table of Contents
* [About The Project](#-about-the-project)
* [Project Workflow](#-project-workflow)
* [Getting Started](#-getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#-usage)
* [Results & Performance](#-results--performance)
* [License](#-license)
* [Contact](#-contact)

---

## 🧐 About The Project

The primary goal of this project was to determine if meaningful sentiment clusters could be identified from text embeddings in a purely unsupervised manner. Instead of traditional supervised fine-tuning, this experiment follows a multi-step process:

1.  **Preprocessing**: Cleaning and normalizing the raw text of 50,000 movie reviews.
2.  **Embedding**: Using the pre-trained `roberta-base` model from Hugging Face to generate high-dimensional vector representations (embeddings) of each review.
3.  **Dimensionality Reduction**: Training a neural network Autoencoder to compress the 768-dimension RoBERTa embeddings into a dense 64-dimension latent space.
4.  **Clustering**: Applying the K-Means algorithm to the compressed embeddings to group the reviews into two clusters, representing positive and negative sentiment.
5.  **Evaluation**: Measuring the quality of the clusters against the true labels using metrics like Silhouette Score, Adjusted Rand Index, and clustering accuracy.

**Key Technologies Used:**
* Python
* PyTorch
* Hugging Face `transformers`
* `scikit-learn`
* `pandas` & `numpy`
* `symspellpy` for spell correction

---

## ⚙️ Project Workflow

The architecture and data flow of the project can be visualized as follows:

1. Data Loading and Preprocessing 🧹
The process begins by preparing the raw text data from the IMDB dataset for the language model.
Load Data: The IMDB Dataset.csv file, containing 50,000 movie reviews, is loaded into a pandas DataFrame. The review text and sentiment labels are separated.
Clean Text: Several cleaning functions are applied sequentially to each review:
HTML Tag Removal: Strips out HTML tags like <br />.
Chat Slang Conversion: Expands common internet acronyms (e.g., "LOL" becomes "Laughing Out Loud") to their full form.
Spell Correction: A significant step where the symspellpy library is used to correct spelling mistakes. The notebook applies this in a multi-pass process to improve text quality.
URL and Case Normalization: Removes any URLs and converts all text to lowercase for consistency.

2. Feature Extraction with RoBERTa 🤖
Once the text is clean, the pre-trained RoBERTa model is used to convert each review into a high-dimensional numerical vector (embedding).
Model Loading: The roberta-base model and its corresponding tokenizer are loaded from the Hugging Face transformers library.\
Embedding Generation: Each preprocessed review is passed through the RoBERTa model. The model outputs a vector for each token in the review.
CLS Token Extraction: For each review, the embedding of the special [CLS] token is extracted. This 768-dimensional vector serves as a semantic summary of the entire review's content.

3. Dimensionality Reduction via Autoencoder 📉
The 768-dimensional embeddings are dense and computationally heavy. An Autoencoder is trained to compress these vectors into a smaller, more manageable latent space while preserving essential information.
Architecture: A PyTorch-based Autoencoder is built with an encoder that maps the input from 768 → 512 → 256 → 64 dimensions and a decoder that reconstructs it back to 768 dimensions.
Training: The Autoencoder is trained for 15 epochs on the RoBERTa embeddings. Its goal is to minimize the reconstruction error (Mean Squared Error), forcing the 64-dimensional bottleneck layer to learn a rich, compressed representation of the data.
Transformation: The trained encoder is then used to convert all 50,000 review embeddings from 768 dimensions into 64-dimensional latent vectors.

4. Unsupervised Clustering with K-Means 📊
With the reviews now represented as dense 64-dimensional vectors, the K-Means algorithm is used to group them without using the original sentiment labels.
Clustering: K-Means is configured to find two clusters (k=2), hypothesizing that these will correspond to positive and negative sentiments.
Optimization: The algorithm is run 10 times with different random starting points, and the best result is selected based on the one that yields the highest Silhouette Score (0.6121), indicating the best-defined clusters.

5. Evaluation and Visualization 📈
Finally, the quality of the unsupervised clusters is assessed by comparing them against the true labels and visualizing the results.
Visualization: PCA is used to further reduce the 64-dimensional latent vectors into 2 dimensions, allowing for a 2D scatter plot visualization of the two clusters.



---

## 🚀 Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

Make sure you have Python 3.8+ and `pip` installed on your system. A GPU is highly recommended for generating the embeddings.

### Installation

It is a ipynb file, so run it in jupyter notebook or kaggle notebook, no installation required, unless you want to do it in VS code which would require environment creation and dependency installation.

---

## 💻 Usage

To run the full pipeline, execute the main Jupyter Notebook:

```sh
jupyter notebook "imdb-review.ipynb"


📊 Results & Performance
After clustering the compressed embeddings, the model's ability to separate sentiments was evaluated. The K-Means algorithm achieved a Best Silhouette Score of 0.6121, indicating a reasonably good separation between the generated clusters.

The clusters were then visualized by reducing the 64-dimensional latent vectors to 2D using PCA.

When comparing the cluster assignments to the actual sentiment labels, the performance was as follows:

Metric	Score
Silhouette Score (Latent)	0.6121
Silhouette Score (PCA 2D)	0.7262
Adjusted Rand Index	0.0002
Clustering Accuracy	0.5076
While the silhouette score is promising, the low Adjusted Rand Index and near-random accuracy (50.76%) suggest that while the model found distinct clusters, these clusters do not perfectly align with the human-annotated positive/negative sentiment labels. This highlights the challenge of unsupervised sentiment discovery.

📝 License
This project is distributed under the MIT License. See the LICENSE file for more information.

📫 Contact
Your Andalib Iftakher - andalib.iftakher138@gmail.com

Project Link: https://github.com/dev-andalib/LLM-RoBERTa
