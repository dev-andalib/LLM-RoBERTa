# Unsupervised Sentiment Analysis of IMDB Reviews using RoBERTa & K-Means

![Project Banner](https://i.imgur.com/3Z6gJ4h.png)

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

![Workflow Diagram](https://i.imgur.com/eBwzY3v.png)

---

## 🚀 Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

Make sure you have Python 3.8+ and `pip` installed on your system. A GPU is highly recommended for generating the embeddings.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    cd your_repository_name
    ```
2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install the required packages from `requirements.txt`:**
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: You can generate the `requirements.txt` file in your local environment by running `pip freeze > requirements.txt`)*

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
