# Hybrid-BERT-RNN-Model-for-Emotion-Recognition-in-Text
This project implements a hybrid BERT-RNN model for emotion recognition in text, combining the strengths of BERT for contextual understanding and a RNN with LSTM layers for sequential data processing. The model is trained on a dataset to classify text into six emotions: sadness, joy, love, anger, fear, and surprise.
The Jupyter notebook Hybrid_BERT_RNN_Model_for_Emotion_Recognition_in_Text.ipynb contains the complete implementation, including data preprocessing, model architecture, training, evaluation, and interactive prediction functionalities. Visualizations such as F1-score bar charts, emotion distribution pie charts, and precision-recall-F1 plots are also included to analyze model performance.
# Features
- Hybrid Model: Integrates BERT for contextual embeddings with LSTM and attention mechanisms for enhanced sequence modeling.
- Dataset: Utilizes a dataset (e.g., from Hugging Face) with labeled text for six emotions.
- Data Augmentation: Includes augmentation techniques to improve model robustness.
- Focal Loss: Employs Focal Loss to handle class imbalance in the dataset.
- Interactive Prediction: Allows users to input text and predict emotions interactively.
- Visualizations: Provides plots for F1-scores, emotion distribution, and precision-recall metrics.
# Requirements
To run the notebook, you need the following dependencies:
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Datasets (Hugging Face)
- NLTK (for text preprocessing, if applicable)
# Usage
- Open the Notebook: Launch Jupyter Notebook or JupyterLab:
- Run the Cells:
  - Execute the cells in order to set up the environment, load the dataset, train the model, and evaluate its performance.
  - The notebook includes a section for interactive prediction where you can input text to classify emotions.
- Interactive Prediction: After training, the final cell allows you to input text (up to 10 times) to predict emotions. Type exit to stop. Example:
- Visualizations: The notebook generates three plots:
  - Bar chart of F1-scores by emotion.
  - Pie chart of emotion distribution in the test set.
  - Line plot comparing precision, recall, and F1-scores across emotions. These are saved as f1_scores_by_emotion.png, emotion_distribution.png, and precision_recall_f1.png.
# Model Architecture
- BERT: Pre-trained bert-base-uncased model for generating contextual embeddings.
- LSTM: Bi-directional LSTM with configurable hidden dimensions and layers to capture sequential patterns.
- Attention Mechanism: Multi-head attention to focus on important parts of the sequence.
- Classifier: Fully connected layers with dropout for emotion classification.
# Training and Evaluation
- Hyperparameters:
  - MAX_LEN: Maximum sequence length for tokenization.
  - BATCH_SIZE: Batch size for training and evaluation.
  - EPOCHS: Number of training epochs.
  - LEARNING_RATE: Learning rate for the AdamW optimizer.
  - LSTM_HIDDEN_DIM, LSTM_LAYERS, ATTENTION_HEADS, DROPOUT: Configurable model parameters.
- Loss Function: Focal Loss to address class imbalance.
- Evaluation Metrics: Precision, recall, and F1-score for each emotion class, visualized in the notebook.
# Results

The model achieves the following F1-scores on the test set:
- Sadness: 0.74
- Joy: 0.42
- Love: 0.60
- Anger: 0.66
- Fear: 0.53
- Surprise: 0.72
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The emotion distribution in the test set is visualized in a pie chart, showing the sample count for each emotion.
# Notes
- Training Time: Training may take significant time depending on hardware (GPU recommended).
- Class Imbalance: The dataset shows an imbalance (e.g., 695 samples for surprise vs. 66 for fear), addressed by Focal Loss.
- Interruptions: The provided notebook output indicates a KeyboardInterrupt during training. Ensure sufficient computational resources and run the training cell uninterrupted.
- Model Saving: The best model is saved as best_model.pt during training for later use.
# License
This project is licensed under the MIT License. See the LICENSE file for details.
