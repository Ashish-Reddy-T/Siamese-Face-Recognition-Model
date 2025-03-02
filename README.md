# Face Recognition from Device's Webcam

This project implements a **Siamese Neural Network** for one-shot face recognition, inspired by the [Siamese Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) paper. It includes real-time face verification via a webcam and a Kivy-based GUI.

---

## Features
- **One-shot Learning**: Recognize faces with just one training example per person.
- **Data Augmentation**: Synthetic data generation (brightness/contrast adjustments, flips, JPEG compression).
- **Siamese Network**: Custom `L1Dist` layer for similarity measurement.
- **Real-Time Verification**: Kivy GUI for live webcam verification.
- **Pre-trained Model**: `siameseModelV2.h5` included for immediate use.
- **Metrics**: Precision, Recall, and adjustable verification thresholds.

---

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Kivy
- OpenCV
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install tensorflow kivy opencv-python numpy matplotlib
```
Clone the repository:
```
git clone https://github.com/Ashish-Reddy-T/Siamese-Face-Recognition-Model.git
```
Navigate into the directory and you'll have it:
```
cd Siamese-Face-Recognition-Model
```

---

## Project Structure

```
.
├── data/                          # Training datasets
│   ├── anchor/                    # Anchor images (1 image per person)
│   ├── positive/                  # Positive matches
│   └── negative/                  # Negative matches
├── application_data/              # Verification data
│   ├── input_image/               # Webcam-captured input
│   └── verification_images/       # Reference database
├── layers.py                      # Custom L1 distance layer
├── faceid.py                      # Kivy application
├── siameseModelV2.h5              # Pre-trained model
└── training_checkpoints/          # Training checkpoints (auto-generated)
```

---

## Training the Model

### 1. Data Preperation
- Add anchor images to `data/anchor/`.
- Add positive/negative samples to `data/positive/` and `data/negative/`.
  
### 2. Data Augmentation

Run the augmentation script (included in the training code):
```
# Augments anchor images with brightness/contrast/flips/saturation
augmented_images = data_aug(img_tensor)
```

### 3. Training

Train the Siamese network
```
# Train for 50 epochs
train(train_data, EPOCHS=50)
```
- Checkpoints saved to `training_checkpoints/`.

### 4. Evaluation

```
# Calculate precision and recall
precision = Precision()
recall = Recall()
print(f"Precision: {precision.result().numpy()}, Recall: {recall.result().numpy()}")
```

---

## Running the Application

### 1. Prepare Verification Database
Add reference images to `applications_data/verification_images/`.

### 2. Launch the GUI
```
python faceid.py
```

### 3. Real-Time Verification
1. Position your face in the webcam feed (crop area: 250px-500px vertically, 600px-850px horizontally).
2. Click __Verify__ to compare against the database.
3. Results appear as __"Verified"__ or __"Unverified"__.

---

## Customization

### Adjust Thresholds
Modify in faceid.py:
```
detection_threshold = 0.5      # Minimum confidence for a positive match
verification_threshold = 0.8   # Fraction of positive matches required
```

### Model Architecture
Edit `make_embedding()` in the training script:
```
def make_embedding():
    inp = Input(shape=(105, 105, 3))
    c1 = Conv2D(64, (10,10), activation='relu')(inp) # Modify layers here
    ...
```

---

## Results

- __Precision__: ~95%
- __Recall__: ~93%
- __Verification Accuracy__: ~85% (with current thresholds)

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

Distributed under the MIT License.

---

## Acknowledgments

- Original research: [Siamese Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).
- Special thanks to [Nicholas Renotte](https://www.youtube.com/@NicholasRenotte) for their guidance via a YouTube tutorial.
- Built with [TensorFlow](https://www.tensorflow.org/), [Kivy](https://kivy.org/), and [OpenCV](https://opencv.org/).

---
