# **YouTube Content Guidelines Analyzer**  

🚀 **A comprehensive tool to analyze YouTube titles, descriptions, and thumbnails for policy compliance, demonetization risks, and content violations.**  

---

## **📌 Table of Contents**  
1. [Features](#-features)  
2. [Installation](#-installation)  
3. [Usage](#-usage)  
4. [Configuration](#-configuration)  
5. [How It Works](#-how-it-works)  
6. [Contributing](#-contributing)  
7. [License](#-license)  

---

## **✨ Features**  
✅ **Text Analysis**  
- Scans titles & descriptions for policy violations  
- Detects **hate speech, violence, harassment, sexual content, misinformation, and copyright issues**  
- Provides **severity ratings** and **suggested fixes**  

✅ **Sentiment Analysis**  
- Uses **TextBlob** to determine if content is **positive, negative, or neutral**  

✅ **Image Analysis (Thumbnails)**  
- **OCR** to extract text from images  
- **Skin detection** to flag potential nudity  
- Checks for inappropriate visuals  

✅ **Machine Learning (LSTM Model)**  
- Predicts **risk scores** based on historical data  
- Improves accuracy over time  

✅ **YouTube API Integration**  
- Analyze **existing videos** by ID  
- Fetch metadata for deeper insights  

✅ **Multilingual Support**  
- Supports **non-English content** via Google Translate  

✅ **Historical Trend Tracking**  
- Tracks **risk scores** over time  
- Visualizes compliance trends  

✅ **User-Friendly GUI**  
- Built with **Tkinter**  
- Easy-to-use interface for non-technical users  

---

## **⚙️ Installation**  

### **Prerequisites**  
- Python 3.8+  
- Tesseract OCR (for image text extraction)  

### **Step 1: Clone the Repository**  
```bash
git clone https://github.com/yourusername/youtube-guidelines-analyzer.git
cd youtube-guidelines-analyzer
```

### **Step 2: Install Dependencies**  
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### **Step 3: Install Tesseract OCR**  
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)  
- **Mac**:  
  ```bash
  brew install tesseract
  ```  
- **Linux (Debian/Ubuntu)**:  
  ```bash
  sudo apt install tesseract-ocr
  ```  

### **Step 4: Get a YouTube API Key**  
1. Go to [Google Cloud Console](https://console.cloud.google.com/)  
2. Create a project & enable **YouTube Data API v3**  
3. Generate an **API key**  
4. Replace `YOUR_API_KEY` in `main.py`  

---

## **🚀 Usage**  

### **1. Run the Application**  
```bash
python main.py
```

### **2. GUI Workflow**  
- **Enter** video **Title** and **Description**  
- **Upload** a thumbnail (optional)  
- Click **"Analyze"**  
- View **detailed report**  

### **3. Command-Line Options**  
```bash
# Analyze text only
python main.py --title "Your Video Title" --desc "Your Description"

# Analyze a YouTube video by ID
python main.py --video-id "VIDEO_ID"

# Analyze non-English text
python main.py --text "Foreign Text" --lang "es"
```

---

## **🔧 Configuration**  

### **1. Custom Keywords & Triggers**  
Edit `demonetization_triggers.csv` to add/remove terms.  

### **2. Retrain ML Model**  
```bash
python train_model.py --data "your_dataset.csv"
```

### **3. Change Language Support**  
Modify `target_lang` in `translate_and_analyze()` for different languages.  

---

## **🤖 How It Works**  
1. **Text Analysis**  
   - Uses **regex & NLP** to detect violations  
   - Checks **sentiment** (positive/negative/neutral)  

2. **Image Analysis**  
   - **OCR** extracts text  
   - **OpenCV** detects skin tones  

3. **Machine Learning**  
   - **LSTM model** predicts risk scores  
   - Improves with **user feedback**  

4. **Historical Trends**  
   - Stores past analyses  
   - Plots **risk trends** over time  

---

## **🤝 Contributing**  
1. Fork the repo  
2. Create a branch (`git checkout -b feature/new-feature`)  
3. Commit changes (`git commit -m "Add new feature"`)  
4. Push (`git push origin feature/new-feature`)  
5. Open a **Pull Request**  

---

## **📜 License**  
This project is licensed under **MIT License**.  

---

## **📊 Example Output**  
![GUI Screenshot](https://)  

---

### **💡 Need Help?**  
Open an **Issue** or reach out at **harshavardhankarne@gmail.com**.  

🚀 **Happy Analyzing!** 🚀
