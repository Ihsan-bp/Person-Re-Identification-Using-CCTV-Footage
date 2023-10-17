# Person-Re-Identification-Using-CCTV-Footage
<h2>Introduction</h2>
This project's goal is to identify individuals across multiple camera views in publicly available CCTV footage. It covers various steps, from data collection and preprocessing to person detection, tracking, feature extraction, and re-identification model development.

<h2>Table of Contents</h2>
1.	Data Collection and Preprocessing
2.	Person Detection and Tracking
3.	Feature Extraction
4.	Person Re-Identification Model
5.	Visualization and Demonstration

<h2>Step 1: Data Collection and Preprocessing</h2>
<h3>Data Collection</h3>
•	<b>Dataset:</b> The dataset used for this project is Market-1501, a comprehensive dataset for person re-identification.


•	<b>Dataset Description:</b> Market-1501 contains 1501 identities captured by six different cameras, comprising 32,668 pedestrian image bounding-boxes. Each person has an average of 3.6 images from multiple viewpoints.

•	<b>Train-Test Split:</b> The dataset is divided into two parts - 750 identities are used for training, while the remaining 751 identities are reserved for testing.

•	<b>Testing Protocol:</b> The official testing protocol involves using 3,368 query images as a probe set to find the correct match among 19,732 reference gallery images.

<h3>Data Preprocessing</h3>
•	<b>Dataset Preprocessing:</b> To make the dataset suitable for model training, it underwent the following preprocessing steps:

•	<b>Loading Images:</b> Images were loaded from the dataset using appropriate libraries, e.g., PyTorch, timm (PyTorch Image Models), Pandas, NumPy, Matplotlib, scikit-learn, Albumentations, OpenCV and TQDM.

•	<b>Resizing:</b> Images were resized to a consistent format (e.g., 224x224) to ensure uniformity.

•	<b>Normalization:</b> The pixel values of images were normalized to the [0, 1] range for model compatibility.

<h2>Step 2: Person Detection and Tracking</h2>
<h3>Person Detection</h3>
•	<b>Object Detection Model:</b> For the task of person detection, we used EfficientNet, a state-of-the-art image classification model. While our primary focus is on person re-identification, EfficientNet helps extract relevant features from the input images, which indirectly contributes to person detection.

<h3>Tracking Algorithm</h3>
•	Tracking Algorithm: To effectively track individuals across frames and camera views, we implemented a custom tracking algorithm. This algorithm assigns unique IDs to individuals detected in each frame and maintains these IDs as people move across different frames and camera viewpoints.

<h3>Sample Outputs</h3>
For a practical demonstration of the person detection and tracking processes, you can refer to our project's outputs and visualizations. These outputs showcase the ability of the tracking system to follow individuals effectively across different frames and camera views.

<h2>Step 3: Feature Extraction</h2>
In this step, we extract relevant features from detected and tracked individuals. Effective feature extraction plays a critical role in person re-identification.
<h3>Feature Extraction Methods</h3>
We have employed the following feature extraction methods:

<b>1.	CNN Embeddings:</b> To extract high-level features from the detected and tracked individuals, we used Convolutional Neural Network (CNN) embeddings. In particular, we employed an EfficientNet model pre-trained on a large dataset. These embeddings capture the distinctive characteristics of each person and are used for subsequent re-identification tasks.

<b>2.	Color Histograms:</b> In addition to CNN embeddings, we also considered color histograms to capture color information. However, the primary emphasis is on CNN embeddings due to their effectiveness in re-identification.

<h2>Step 4: Person Re-Identification Model</h2>
In this step, we design and implement a person re-identification model using PyTorch. The re-identification model leverages features extracted in the previous step and is trained on our dataset for person re-identification tasks.
<h3>Model Architecture</h3>
The person re-identification model is implemented using PyTorch, with the following architecture:

•	<b>Backbone Model:</b> We employed an EfficientNet-based architecture as the backbone model, which has demonstrated effectiveness in various computer vision tasks. The backbone model is pre-trained on a large dataset to capture useful features.

•	<b>Classification Head:</b> To adapt the backbone model for re-identification, we modified the classifier head by replacing it with a custom linear layer. This new layer maps the extracted features to a lower-dimensional space, making it suitable for re-identification tasks.

<h3>Training Process</h3>
The training process consists of the following key components:

•	<b>Loss Function:</b> We used the Triplet Margin Loss as our loss function. This loss encourages the model to minimize the distance between positive pairs (images of the same person) while maximizing the distance between negative pairs (images of different individuals).

•	<b>Optimizer:</b> We employed the Adam optimizer with a learning rate of 0.001 to train the model.

•	<b>Training Data:</b> The training dataset consists of labeled samples containing anchor, positive, and negative images.

•	<b>Epochs:</b> The model was trained over multiple epochs. Specifically, it was trained for 15 epochs in our experiment.

<h2>Step 5: Visualization and Demonstration</h2>
In this step, we aim to showcase the effectiveness of our person re-identification model through visualizations and demonstrations. The visualizations help us understand the model's capabilities in accurately re-identifying individuals across different camera views.

<h2>Visualizations</h2>
We provide visualizations to illustrate the model's performance and effectiveness. These visualizations can include:

•	<b>Sample Re-Identification Results:</b> Display images of persons successfully re-identified by the model. Showcases where the model accurately matches an individual across different camera views.

•	<b>Matching Confidence Scores:</b> Visualize the confidence scores or similarity scores generated by the model for each match. This helps understand how certain the model is about its re-identification results.

<h3>Demonstrations</h3>
We demonstrate how our model can accurately re-identify individuals across different camera views. Here's an overview of the demonstrations:

•	<b>Cross-Camera Re-Identification:</b> We showcase how our model accurately re-identifies individuals as they move from one camera view to another. This demonstrates the practicality and robustness of our solution.

•	<b>Re-Identification Across Time:</b> We demonstrate the model's ability to re-identify individuals over time. This is especially important in real-world scenarios where individuals may appear at different times.

<h3>sample  visualization</h3>






