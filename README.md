# Trash Classification Model

This repository contains code to train a deep learning model for classifying trash into categories. It leverages transfer learning with ResNet and supports data augmentation, class balancing, and automated development processes using GitHub Actions.

## Dataset

The dataset used is the [TrashNet dataset](https://huggingface.co/datasets/garythung/trashnet), which contains images of trash sorted into six categories: `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.

1. Download the dataset from [Hugging Face TrashNet](https://huggingface.co/datasets/garythung/trashnet).
2. After downloading, extract and organize it in the following structure:

   ```plaintext
   notebooks/my-trash-dataset/train/
   ├── cardboard
   ├── glass
   ├── metal
   ├── paper
   ├── plastic
   └── trash

# Setup 
1. Clone the repository
   ```git
   git clone https://github.com/your-username/trash-classification cd trash-classification-model

2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment:
Ensure that the dataset is in the path notebooks/my-trash-dataset/train/.
(Optional) Configure cloud access by setting your credentials if training on Google Cloud, AWS, etc


# Automated Model Training with GitHub Actions
This repository includes a GitHub Actions workflow for automating the model training process. Upon each push, the workflow:

Sets up the environment and installs dependencies.
Downloads the dataset from cloud storage (if configured in download_dataset.py).
Trains the model and logs results to Weights & Biases.
To activate the workflow:

Set up secrets in GitHub for any cloud storage credentials or WandB API keys.

# Results and Model Evaluation
After training, the model's performance is evaluated using accuracy and loss metrics on both the training and validation datasets. Here is a summary of the results:

1. **Final Accuracy and Loss**:
   - Training Accuracy: 90.5%
   - Validation Accuracy: 84.15%
   - Training Loss: 0.25
   - Validation Loss: 0.50
  
2. **Confusion Matrix**:
   -  A confusion matrix was created to illustrate the model's performance across different trash categories. You can visualize it in the notebook `model-train.ipynb` under the evaluation section.
   - This matrix highlights correct and incorrect predictions for each class, allowing a deeper insight into which classes may need more data or fine-tuning.

    ![1730033556545](https://github.com/user-attachments/assets/4ef82a2e-8609-422a-a337-aecaf266b721)

   * Cardboard: Predicted correctly 15 times but misclassified frequently as "paper" (28) and "glass" (16).
   * Glass: This class performed better than others, with 32 correct predictions, but it was also misclassified, mostly as "paper" and "cardboard."
   * Metal: Misclassified heavily as "paper" (20) and "plastic" (15), showing lower classification accuracy for this class.
   * Paper: Predicted correctly 25 times, but also misclassified as "glass" (29) and "plastic" (22).
   * Plastic: This class is often confused with "paper" and "metal," with 20 correct predictions but significant misclassifications.
   * Trash: This class has the fewest correct predictions (7), with frequent misclassifications across all other classes. Given that this class had the least amount of data, it may benefit from data augmentation or additional samples.
  
   The "trash" class has a lower correct classification rate, likely due to fewer samples in the training set. This impacts the model’s ability to accurately classify instances of "trash" and is reflected in its distribution across various misclassified categories.

3. **Learning Curve**
   ## Validation Accuracy
     ![1730033850859](https://github.com/user-attachments/assets/55688c59-7240-44e7-9181-0cea4975e2b3)

   ## Validation Loss
     ![1730033863330](https://github.com/user-attachments/assets/192c6272-12fc-4daf-98ce-5eb3e8bf01ac)

   ## Training Accuracy
     ![1730033877267](https://github.com/user-attachments/assets/fed26208-cc49-4844-9c43-0733c7a9ffe7)

   ## Training Loss
      ![1730033885805](https://github.com/user-attachments/assets/cf6d99b8-92a8-49e0-848a-a04796108e59)

# Overall
The model appears to be fairly reliable in predicting most classes, but certain classes still exhibit higher misclassification rates. This could be due to an imbalanced dataset or the visual similarity between certain classes.



    
