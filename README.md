# OPTIMISED SKIN CANCER DETECTION USING AUTO ENCODER AND BIO-INSPIRED NEURAL NETWORKS


## ABOUT:
The optimised skin cancer detection system shown in this project combines a Bio-Inspired Neural Network (BINN) classifier with Autoencoder-based feature extraction.  Enhancing diagnostic precision, lowering computing overhead, and enabling reliable categorisation of dermoscopic pictures into benign and malignant groups are the objectives.

 The method makes effective and accurate predictions by using a biologically inspired classification mechanism and unsupervised deep learning to acquire meaningful latent representations of skin lesions.  The system intends to assist dermatologists, lessen the manual diagnostic burden, and encourage early melanoma identification by incorporating these models.

## FEATURES:

* Image-Based Diagnosis: Works directly with dermoscopic lesion images for classification.

* Autoencoder Feature Extraction: Learns compact, noise-free latent features from raw images.

* Bio-Inspired Neural Network (BINN): Utilizes biologically inspired principles (e.g., spiking networks / reservoir computing) for robust classification.

* Dimensionality Reduction: PCA/UMAP improves model speed and performance by reducing redundant features.

* Interpretable Output: Provides class labels, confidence scores, and optional visualization of important lesion regions.

* High Efficiency: Lightweight architecture enables fast inference on regular hardware.

* Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC.

* Scalable Design: Can be extended to multi-class skin lesion prediction in future work.

## REQUIREMENTS:

Operating System: Windows 10/11 (64-bit), Ubuntu Linux, or Google Colab environment

Development Environment: Python 3.7+

Machine Learning & Deep Learning Frameworks: TensorFlow / Keras,PyTorch,Scikit-Learn

Data & Processing Libraries: NumPy, Pandas, OpenCV, Matplotlib, Seaborn, SHAP (optional for explainability)

Version Control: Git

IDE / Notebook:Google Colab, Jupyter Notebook 

Deep Learning Models – Autoencoder & BINN

Optimization Algorithms – PSO / GA

## SYSTEM ARCHITECTURE:

The optimized skin cancer detection system consists of five major components:

1.Autoencoder

Extracts compressed latent features from dermoscopic images

Removes noise and highlights important skin lesion patterns

2.Feature Reduction Module (PCA/UMAP)

Reduces dimensionality of encoded vectors

Enhances model generalization and efficiency

3.Bio-Inspired Neural Network (BINN)

Classifies images into benign or malignant

Inspired by biological neural processing for improved robustness

4.Prediction & Interpretation Layer

Displays diagnosis with confidence score

Optional visualization to highlight relevant lesion regions

5.Evaluation Module

Generates Accuracy, Precision, Recall, F1-Score, ROC-AUC

Compares predicted vs actual outcomes with visual graphs


<img width="1024" height="585" alt="image" src="https://github.com/user-attachments/assets/bfbbc0fe-4612-4db1-ad5c-279edbbd44ca" />



## OUTPUT:

<img width="983" height="717" alt="image" src="https://github.com/user-attachments/assets/0a59b783-92b3-4bb7-b68f-3141b8608cdc" />

<img width="1706" height="750" alt="image" src="https://github.com/user-attachments/assets/043ee18f-eeb6-476d-abd3-050e45eb0ec1" />

<img width="691" height="788" alt="image" src="https://github.com/user-attachments/assets/56ac414d-3ee3-4eb1-96ff-df149c518b27" />

<img width="1091" height="763" alt="image" src="https://github.com/user-attachments/assets/83037cdd-b14e-4c30-a35f-392bc1a75ba3" />

<img width="908" height="454" alt="image" src="https://github.com/user-attachments/assets/7a7273c2-6528-47b9-a9d7-fa9c83a6f82d" />

<img width="893" height="684" alt="image" src="https://github.com/user-attachments/assets/ecf04a13-030d-4734-a29a-7122707089cd" />

<img width="1304" height="793" alt="image" src="https://github.com/user-attachments/assets/efce49da-11b3-450b-93dd-cc44f3eb8e1a" />



## RESULTS AND IMPACT:
* Detection Accuracy:

Achieves high accuracy (e.g., 92–96% depending on dataset and preprocessing)

* Clinical Benefits:

Supports dermatologists with fast, preliminary screening

Enables early melanoma detection

Enhances decision-making with interpretable outputs

* Operational Impact:

Fast and lightweight model suitable for clinics, screening centers, and mobile health apps

Reduces dependence on very large training datasets

* Innovation:

Combines unsupervised feature extraction with biologically inspired classification

Outperforms baseline CNN and ANN models in accuracy and processing speed


## ARTICLE PUBLISHED/REFERENCES:
[1] A. Esteva et al., “Dermatologist-level classification of skin cancer with deep neural networks,” Nature, vol. 542, no. 7639, pp. 115–118, Feb. 2017. 


[2] P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Sci. Data / arXiv:1803.10417, 2018. 


[3] N. Gessert, T. Schott, H. A. F. et al., “Skin lesion classification using ensembles of multi-scale deep convolutional neural networks (ISIC 2019 challenge methods),” arXiv:1910.03910, 2019. 


[4] Y. Lu, S. Li, H. Huang, “Anomaly detection for skin disease images using variational autoencoders,” arXiv:1807.01349, 2018. 


[5] C. Magalhães, J. Silva, et al., “Systematic review of deep learning techniques in skin cancer diagnosis,” MDPI — review article, 2024. 


[6] P. Tschandl, C. Rosendahl, H. Kittler, “HAM10000: A large collection of multi-source dermatoscopic images,” PLOS / dataset description, 2018. 


[7] M. Toğaçar, U. Ergen, “Intelligent skin cancer detection applying autoencoder, spiking and convolutional neural networks,” Computers in Biology and Medicine, 2021. 


[8] D. Bardou, T. Zhang, et al., “Hair removal in dermoscopy images using variational autoencoder,” Sensors, 2022.


[9] S. Wang, H. Chen, Y. Zhang, “Bionic artificial neural networks in medical image analysis,” Biomimetics, 2023. 

[10] M. Naqvi, S. Q. Gilani, T. Syed, O. Marques, H.-C. Kim, “Skin cancer detection using deep learning — a review,” Diagnostics, 2023. 

[11] H. Alharbi, A. A. Alzahrani, “Enhanced skin cancer diagnosis: deep feature extraction and classifier fusion,” Scientific Reports / MDPI article (2024). 


[12] S. A. H. Shah, “Explainable AI-based skin cancer detection using CNN and PSO for feature selection,”                    Journal / MDPI, 2024.


[13] O. Salih, “Optimization of Convolutional Neural Network with Genetic Algorithms for automatic skin lesion classification,” Applied Sciences, 2023.


[14] R. Mundada, “Skin Cancer Prediction by Incorporating Bio-inspired Optimization and Deep Features,” Springer / ACM conference paper (2024).

[15] Q. Huang, “A skin cancer diagnosis system for dermoscopy images using particle swarm optimization for feature selection,” Elsevier / journal paper, 2023.

[16] B. Cassidy, N. Codella, et al., “Analysis of the ISIC image datasets: usage, benchmarks and insights,” Computerized Medical Imaging and Graphics, 2021.


[17] N. Gessert, “Skin lesion classification and ISIC challenge method,” IEEE / arXiv (2019) — methods and ensemble approaches. 

[18] M. Combalia, N. C. Codella, V. Rotemberg, et al., “BCN20000: Dermoscopic lesions in the wild,” arXiv preprint (2019).


[19  ]S. Hermosilla, “Skin Cancer Detection and Classification: Systematic Review — deep learning approaches and limitations,” PMC / review (2024).


[20] A. Khan et al., “CAD-Skin: A hybrid CNN-Autoencoder framework for precise detection and classification of skin lesions,” Bioengineering (MDPI), 2025.


[21] S. Vijh, P. Gaurav, H. M. Pandey, “Hybrid bio-inspired algorithm and convolutional neural network for automatic tumor detection,” Springer / conference article (2020).


[22] S. Mandal, “Active learning with particle swarm optimization for medical image classification,” PMC article (2024).


[23] O. Akinrinade, “Skin cancer detection using deep machine learning approaches — survey and experimental study,” ScienceDirect / 2025.


[24] A. Al-Waisy, “A deep learning framework for automated early diagnosis — Skin-DeepNet,” Scientific Reports / Nature (2025).


[25] H. Arshad, “Multiclass skin lesion classification and localization from deep models,” BMC Med Inform Decis Mak, 2025.


[26]  P. Georgiadis, “A case study in skin cancer classification: methods and lessons,” PMC article (2025).

[27] L. Huang, “Survey and reproduction of dermatologist-level skin cancer classification results,” technical report / PDF (2024).


[28]  S. Wang, L. Chen, X. Zhang, “Bio-inspired neural network design for medical image classification,” IEEE Access, 2021. 

[29] H. Lu, X. Li, “Variational Autoencoder and anomaly detection applications in medical images,” arXiv / journal articles (2018–2020).


[30] M. Naqvi, “A comprehensive review: Nature-inspired optimization approaches for feature selection in medical imaging,” Current Medical Imaging, 2020–2024. 
