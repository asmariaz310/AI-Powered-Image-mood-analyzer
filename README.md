# AI-Powered Image Mood Analyzer

## Overview
Mood Analyzer is a real-time mood detection system that uses computer vision and deep learning to analyze facial expressions and determine a person's emotional state. The system provides:

- Real-time mood inference using proportional analysis of detected emotions (angry, disgust, fear, happy, sad, surprise, neutral).
- Mood inference with contextual interpretation  
- Stress level estimation  
- Gender detection  
- Islamic dua (prayer) and references based on detected mood  
- Detailed PDF reports with mood analytics  

## Key Features

- **Multi-source input:** Works with both webcam and recorded video files  
- **Advanced emotion analysis:** Uses EMA (Exponential Moving Average) for smooth predictions  
- **Contextual mood interpretation:** Goes beyond basic emotions to infer moods like "Euphoric" or "Frustrated"  
- **Islamic integration:** Provides relevant duas, Quran references, and Hadith based on detected mood  
- **Comprehensive reporting:** Generates PDF reports with mood distribution and timeline  
- **Stress detection:** Estimates stress levels based on facial cues  
- **Gender detection:** Identifies gender from facial features  

## Technical Details

- **Deep Learning Model:** Custom-trained CNN for emotion recognition  
- **Computer Vision:** OpenCV for face detection and feature extraction  
- **Data Visualization:** Matplotlib for mood analytics  
- **Report Generation:** FPDF for creating detailed PDF reports  
- **Preprocessing:** Includes contrast enhancement, denoising, and edge detection  

## How to Use

You will be prompted to select an input source:
- Webcam (default camera)
  
![image](https://github.com/user-attachments/assets/a10924f9-70ef-4285-9eea-7e666327583a)

- Recorded video file
  
![image](https://github.com/user-attachments/assets/c6bbc062-7244-42d8-a241-f1dca7724b40)

### Controls

- Press **'Q'** to quit the application and generate a report  
- The system will automatically save a recording of your session  

## Output

The system generates:

### Real-time display with:
- Detected face bounding box  
- Current emotion label  
- Mood meter visualization  
- Relevant Islamic dua and references  
- Stress level estimation  
- Gender detection  

### PDF report containing:
- Session information  
- Mood distribution pie chart

![mood_pie](https://github.com/user-attachments/assets/f155b7bc-4ac9-4936-9e73-943a63f0a005)

- Mood timeline visualization

![mood_over_time](https://github.com/user-attachments/assets/77d12b1d-d326-4dfd-b496-832b764ab76d)

- Detailed dua and references  
- Mood trivia  

## Model Architecture

The emotion recognition model is a convolutional neural network (CNN) with:
- **Input layer:** 48×48 grayscale images  
- **Multiple convolutional and pooling layers**  
- **Fully connected layers** for classification  
- **Softmax output** with 7 emotion classes  

## File Structure
- mood-analyzer/
- ├── Data/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Pre-trained models and configurations
- │   ├── model.json&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# CNN model architecture for emotion detection
- │   ├── model.weights.h5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Trained weights for emotion detection model
- │   ├── Gender_deploy.prototxt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Caffe model config for gender classification
- │   └── Gender_net.caffemodel&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Pre-trained weights for gender classification
- ├── Mood_Graphs/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Output directory for mood graphs and PDF reports
- ├── recordings/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Stores video recordings of user sessions
- ├── mood_analyzer.ipynb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Jupyter notebook for running and analyzing the system interactively
- ├── emotion_detection.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Core script for face detection and emotion classification
- ├── app.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Flask server script for running the web interface
- ├── templates/
- │   └── index.html&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# HTML template for the web UI
- ├── Test_videos/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Sample recorded videos for testing
- └── README.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Project documentation file


## Ethical Considerations

- This system should be used responsibly  
- Mood detection technology has limitations and may not always be accurate

## License

This project is licensed under the MIT License.
