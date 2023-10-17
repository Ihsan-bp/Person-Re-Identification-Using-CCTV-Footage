# Person-Re-Identification-Using-CCTV-Footage
<h2>Introduction</h2>
This project's goal is to identify individuals across multiple camera views in publicly available CCTV footage. It covers various steps, from data collection and preprocessing to person detection, tracking, feature extraction, and re-identification model development.

<h2>Table of Contents</h2>
1.	Data Collection and Preprocessing
2.	Person Detection and Tracking
3.	Feature Extraction
4.	Person Re-Identification Model
5.	Visualization and Demonstration

<h3>Step 1: Data Collection and Preprocessing</h3>
<h3>Data Collection</h3>
•	<b>Dataset:</b> The dataset used for this project is Market-1501, a comprehensive dataset for person re-identification.
•	<b>Dataset Description:</b> Market-1501 contains 1501 identities captured by six different cameras, comprising 32,668 pedestrian image bounding-boxes. Each person has an average of 3.6 images from multiple viewpoints.
•	<b>Train-Test Split:</b> The dataset is divided into two parts - 750 identities are used for training, while the remaining 751 identities are reserved for testing.
•	<b>Testing Protocol:</b> The official testing protocol involves using 3,368 query images as a probe set to find the correct match among 19,732 reference gallery images.

<h3>Data Preprocessing</h3>
•	<b>Dataset Preprocessing:</b> To make the dataset suitable for model training, it underwent the following preprocessing steps:
•	<b>Loading Images:</b> Images were loaded from the dataset using appropriate libraries, e.g., [mention_library_used].
•	<b>Resizing:</b> Images were resized to a consistent format (e.g., 224x224) to ensure uniformity.
•	<b>Normalization:</b> The pixel values of images were normalized to the [0, 1] range for model compatibility.

