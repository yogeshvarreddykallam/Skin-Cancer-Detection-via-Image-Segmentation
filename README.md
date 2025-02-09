### **Project Overview: Segmentation and Classification of Skin Cancer Using Machine Learning**

This project aims to develop an automated system for detecting and classifying skin cancer using image processing and machine learning techniques. The system enhances diagnostic accuracy and efficiency compared to traditional manual methods used by dermatologists.

---

## **1. Introduction & Motivation**
Skin cancer is one of the most common and dangerous types of cancer, with early detection being crucial for effective treatment. The traditional method of visual examination by dermatologists can be subjective, time-consuming, and prone to errors. The proposed system addresses these challenges by automating the diagnosis process using **computer vision** and **machine learning algorithms**.

---

## **2. Existing vs. Proposed System**
### **Existing System**
- Relies on **manual visual inspection** by dermatologists.
- Subject to **human errors and inconsistencies**.
- Time-consuming and requires **expert knowledge**.
- Limited **scalability** for mass screenings.

### **Proposed System**
- Uses **image segmentation** to detect regions of interest (skin lesions).
- Employs **feature extraction techniques** to analyze characteristics like shape and texture.
- Trains a **machine learning model** (One-Class Support Vector Machine - SVM) for classification.
- Ensures **early detection, improved accuracy, and consistency** in diagnosis.
- Can be used in **remote and resource-limited healthcare settings**.

---

## **3. Technical Approach**
The system follows a pipeline consisting of multiple steps, each leveraging **computer vision and machine learning** techniques.

### **3.1 Image Segmentation**
- **Purpose:** Isolate the skin lesion from the background in an image.
- **Method:** Converts the image to the **HSV color space** and applies **color thresholding** to extract regions of interest.

### **3.2 Feature Extraction**
- **Purpose:** Identify and quantify key characteristics of the skin lesion.
- **Method:** Uses **Histogram of Oriented Gradients (HOG)** to extract shape and texture-based features.
- **Benefits:** HOG reduces dimensionality while retaining useful information.

### **3.3 Machine Learning Model (One-Class SVM)**
- **Algorithm:** One-Class **Support Vector Machine (SVM)**
- **Training Data:** Trained on **non-cancerous** skin lesions to learn the normal skin texture.
- **Testing:** The model detects **anomalies** in test images and classifies them as potentially cancerous or non-cancerous.

### **3.4 Anomaly Detection**
- The system identifies lesions that deviate from normal skin patterns.
- Cancerous lesions **significantly differ** in texture and shape, making them detectable via anomaly detection.

---

## **4. Implementation Details**
### **4.1 Algorithm**
1. **Data Acquisition:** Collect images of skin cancer and non-cancerous skin conditions.
2. **Image Segmentation:** Use color-based segmentation in the **HSV color space** to isolate lesions.
3. **Feature Extraction:** Apply **HOG descriptor** to extract shape and texture information.
4. **Model Training:** Train a **One-Class SVM** on non-cancerous skin images.
5. **Testing & Classification:** Test on unseen images and classify lesions as normal or cancerous.

### **4.2 Flowchart**
The general pipeline follows:
1. **Input Image** → 2. **Segmentation** → 3. **Feature Extraction** → 4. **Model Training** → 5. **Classification Output**

---

## **5. Program Implementation (Python Code)**
### **Key Libraries Used:**
- `OpenCV (cv2)` for image processing.
- `skimage` for feature extraction using **HOG descriptors**.
- `scikit-learn (sklearn)` for **SVM model training**.
- `matplotlib` for visualization.

### **Key Code Snippets**
#### **1. Image Segmentation**
```python
def segment_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 30, 0])
    upper_color = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image
```
#### **2. Feature Extraction (HOG)**
```python
def extract_features(image):
    image = resize(image, (64, 64))  
    features = hog(image, pixels_per_cell=(4, 4))  
    return features
```
#### **3. Model Training (One-Class SVM)**
```python
classifier = OneClassSVM(kernel='linear', nu=0.1)  
classifier.fit(X_train_features)
```
#### **4. Testing and Prediction**
```python
y_pred = classifier.predict(X_test_features)
```
- If `y_pred[0] == 1`: **Non-cancerous**
- Else: **Possible skin cancer**

---

## **6. Results & Performance Evaluation**
- **Final Accuracy:** The model was tested on a set of **random images**, and the accuracy was calculated using the `accuracy_score` function.
- **Visualization:** The program displays test images along with classification results.
- **Evaluation Metrics:**
  - **Precision, Recall, and F1-score** used to assess model performance.

### **Observations:**
- The system successfully identifies **cancerous and non-cancerous lesions**.
- The One-Class SVM **performs well** in detecting anomalies.
- **False positives and negatives** exist, suggesting further improvements.

---

## **7. Discussion & Future Improvements**
### **7.1 Performance Evaluation**
- The model effectively detects skin cancer lesions **based on anomalies** in texture.
- **Challenges:** Some non-cancerous lesions with unusual features may be misclassified.
- More **diverse training data** can improve accuracy.

### **7.2 Limitations**
- **False Positives:** Non-cancerous lesions with atypical textures may be misclassified.
- **Limited Dataset:** The model relies on the quality and diversity of training images.
- **Segmentation Sensitivity:** Variability in **lighting and skin tones** can affect segmentation accuracy.

### **7.3 Future Directions**
1. **Improve Feature Extraction:**
   - Use **Deep Learning models (CNNs)** for better accuracy.
   - Explore other descriptors like **Local Binary Patterns (LBP)**.
2. **Enhance Segmentation:**
   - Use **Deep Learning-based segmentation models (U-Net, Mask R-CNN)**.
3. **Expand Training Data:**
   - Include **larger datasets** with diverse skin types and conditions.
4. **Implement Real-Time Application:**
   - Develop a **mobile app** for skin cancer screening.

---

## **8. Conclusion**
This project successfully demonstrates the **potential of AI in medical diagnostics**, specifically for **automated skin cancer detection**. The **One-Class SVM** model, combined with **image processing techniques**, provides a reliable method for **early detection of skin cancer**. With further refinements, this system could become a valuable **clinical decision support tool** for dermatologists, improving the accuracy and accessibility of **skin cancer diagnosis**.

---

## **9. References**
The project references **academic research papers** on skin cancer detection, image segmentation, feature extraction, and machine learning techniques used in medical imaging.
