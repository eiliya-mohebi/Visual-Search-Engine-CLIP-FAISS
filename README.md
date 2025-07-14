# 🖼️ Visual Search Engine with CLIP & FAISS

<div align="center">
  <img src="./demo.gif" alt="Demo""/>
</div>

Ever wanted to find an image using another image? This project lets you do just that. Upload a picture, and the AI-powered engine will instantly find the most visually similar images from a library of thousands.

The entire project runs in a single Google Colab notebook.

## ✨ How It Works

The system operates in two main phases: an offline **Indexing Phase** where it learns the dataset, and a real-time **Search Phase** where it finds similar images.

### 1. Indexing Phase: Understanding the Image Library

#### **🧠 Feature Extraction with CLIP**

The goal is to convert every image into a meaningful numerical representation. We use **CLIP (Contrastive Language-Image Pre-training)** for this critical step.

* **Input Processing:** Each image from our library of 8,000 is first pre-processed. It's resized and normalized to match the specific format the CLIP model was trained on.
* **Vision Transformer (ViT) Architecture:** The image is then fed into CLIP's *Image Encoder*, which is a Vision Transformer. The ViT breaks the image down into a grid of smaller patches (e.g., 16x16 pixels). Each patch is treated like a "word" in a sentence. The model then uses a self-attention mechanism to analyze the relationships between these patches, allowing it to understand the objects, textures, and overall context of the image.
* **Embedding Generation:** The output of this process is a single, dense 512-dimension vector known as an "embedding." This vector encapsulates the high-level semantic content of the image.
* **Normalization:** Crucially, each embedding is normalized to have a unit length (a magnitude of 1). This mathematical step is vital because it allows us to use the dot product (Inner Product) as a direct and efficient measure of **Cosine Similarity**.

#### **⚡️ High-Speed Indexing with FAISS**

Once we have 8,000 embeddings, we need a way to search through them quickly. We use **FAISS (Facebook AI Similarity Search)** to build a searchable index.

* **Index Type (`IndexFlatIP`):** We use a specific FAISS index called `IndexFlatIP`.
    * **`Flat`**: This means the index does not compress or partition the vectors. It stores them in their full, original form. This results in a brute-force or exhaustive search, which is perfectly accurate and extremely fast for datasets of this size (thousands of items).
    * **`IP` (Inner Product):** This tells the index that the metric for similarity will be the inner product (or dot product) between vectors. Because our vectors are normalized, maximizing the inner product is mathematically equivalent to maximizing the cosine similarity, which effectively finds the vectors that are "pointing" in the same direction in the high-dimensional space.
* **Population:** All 8,000 normalized embeddings are added to this index, which is then held in memory, ready for queries.

### 2. Search Phase: Finding a Match in Real-Time

When a user uploads an image via the Gradio interface, the following happens in milliseconds:

1.  **Query Processing:** The uploaded image undergoes the exact same pre-processing and feature extraction steps as the library images, resulting in a single 512-dimension, normalized query embedding.
2.  **FAISS Search:** The query embedding is passed to the `index.search(query_embedding, k=9)` method. FAISS rapidly computes the inner product between the query embedding and all 8,000 embeddings in the index.
3.  **Result Retrieval:** FAISS returns the indices (positions in the original dataset) of the top 9 images with the highest inner product scores.
4.  **Display:** The system retrieves the original images and their corresponding labels using these indices and displays them in the user interface.


## 🚀 Get Started in 3 Steps

You can run this entire project for free in Google Colab.

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/eiliya-mohebi/Visual-Search-Engine-CLIP-FAISS.git
    ```

2.  **Upload to Colab**:
    * Go to [colab.research.google.com](https://colab.research.google.com).
    * Click `File` > `Upload notebook...` and select the `.ipynb` file.

3.  **Run It**:
    * Make sure you're using a **GPU** (`Runtime` > `Change runtime type`).
    * Run all the cells from top to bottom. The last cell will give you a public link to the search engine!

## 🛠️ Key Technologies

* **Python 3**: The core programming language used for the entire project.
* **PyTorch**: The deep learning framework used to load and run the pre-trained CLIP model for inference.
* **Hugging Face `transformers` & `datasets`**: Essential libraries for the AI ecosystem. `transformers` allows us to download and use the CLIP model with just a few lines of code, while `datasets` provides a simple way to download and manage the `imagenette` dataset.
* **FAISS**: The high-performance C++ library (with Python bindings) that powers our similarity search. It's crucial for making the search feel instantaneous.
* **Gradio**: A Python library that makes it incredibly easy to build and share a simple, beautiful web UI for machine learning models.
