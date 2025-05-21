# Important Interview Questions & Answers on TensorFlow Image Classification Project

### 1. What is the purpose of `ImageDataGenerator` in Keras?  
**Answer:**  
It generates batches of tensor image data with real-time data augmentation to help prevent overfitting.

### 2. Why do we use `rescale=1/255.` in the `ImageDataGenerator`?  
**Answer:**  
To normalize pixel values from [0, 255] to [0, 1], which improves model convergence.

### 3. What does `validation_split=0.15` do?  
**Answer:**  
It reserves 15% of the dataset for validation while training on the remaining 85%.

### 4. How does `flow_from_directory` work?  
**Answer:**  
It reads images from a directory organized by class subfolders and generates batches with labels.

### 5. Why is `shuffle=True` used in the training generator?  
**Answer:**  
To ensure the training data is mixed randomly each epoch for better generalization.

### 6. Why freeze layers in the pre-trained model?  
**Answer:**  
To keep learned features intact and only fine-tune later layers to adapt to the new task, saving training time and avoiding overfitting.

### 7. What is transfer learning?  
**Answer:**  
Using a pre-trained model on a large dataset (like ImageNet) and adapting it to a new but related task.

### 8. Why use `GlobalAveragePooling2D` instead of `Flatten`?  
**Answer:**  
It reduces overfitting and model size by averaging spatial features instead of flattening all.

### 9. Why add a `Dropout` layer?  
**Answer:**  
To randomly drop neurons during training to reduce overfitting.

### 10. What is the role of the final Dense layer with `softmax` activation?  
**Answer:**  
To output class probabilities for multi-class classification.

### 11. Why choose `categorical_crossentropy` as the loss function?  
**Answer:**  
Because the task involves multi-class classification with one-hot encoded labels.

### 12. What optimizer is used and why?  
**Answer:**  
Adam optimizer is used for adaptive learning rate and faster convergence.

### 13. Why set a low learning rate (0.0001)?  
**Answer:**  
To fine-tune the pre-trained layers gently without large updates.

### 14. What is the purpose of `ModelCheckpoint` callback?  
**Answer:**  
To save the best model weights during training based on validation loss.

### 15. What does the `EarlyStopping` callback do?  
**Answer:**  
It stops training early if validation loss doesn’t improve for a set number of epochs to avoid overfitting.

### 16. How is the model evaluated during training?  
**Answer:**  
Using validation data and metrics like accuracy and loss.

### 17. How do you predict classes on new images?  
**Answer:**  
Load and preprocess the image, expand dimensions, and call `model.predict()` followed by `argmax`.

### 18. Why resize images to 224x224?  
**Answer:**  
Because VGG16 expects 224x224 input size.

### 19. Why is preprocessing required before prediction?  
**Answer:**  
To scale pixels and normalize input as expected by the pre-trained model.

### 20. What is the difference between training and validation generators?  
**Answer:**  
Training generator applies augmentation and shuffling; validation generator usually does not.

### 21. What does `np.argmax` do in predictions?  
**Answer:**  
It selects the class with the highest predicted probability.

### 22. Why call `model.save()` after training?  
**Answer:**  
To save the trained model for later use or deployment.

### 23. What is the significance of `class_mode='categorical'`?  
**Answer:**  
Indicates labels are one-hot encoded vectors for multi-class classification.

### 24. How does data augmentation help?  
**Answer:**  
It artificially increases dataset diversity and reduces overfitting.

### 25. Why use both `train_generator` and `val_generator` from the same directory?  
**Answer:**  
To split data automatically into training and validation sets based on `subset` parameter.

### 26. How can you visualize training progress?  
**Answer:**  
By plotting accuracy and loss curves over epochs using matplotlib.

### 27. How do you interpret the classification report?  
**Answer:**  
It shows precision, recall, f1-score for each class, helpful for model evaluation.

### 28. What are common metrics used for image classification?  
**Answer:**  
Accuracy, precision, recall, F1-score, and confusion matrix.

### 29. Why might you freeze all but last few layers?  
**Answer:**  
Because initial layers learn generic features; last layers adapt to the new dataset.

### 30. How to handle imbalanced classes in such projects?  
**Answer:**  
Use class weights, oversampling, or more data augmentation for minority classes.

---

