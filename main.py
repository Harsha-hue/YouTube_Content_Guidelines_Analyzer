import re
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from googletrans import Translator
from googleapiclient.discovery import build
from textblob import TextBlob
from PIL import Image
import pytesseract
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class YouTubeGuidelinesAnalyzer:
    def __init__(self):
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_lg")
        self.translator = Translator()
        
        # Initialize ML model
        self.ml_model = self.init_ml_model()
        self.tokenizer = Tokenizer(num_words=10000)
        
        # Initialize GUI
        self.root = tk.Tk()
        self.setup_gui()
        
        # Load databases
        self.violation_categories = self.load_violation_categories()
        self.demonetization_triggers = self.load_demonetization_triggers()
        self.examples_db = self.load_examples_db()
        
        # Initialize YouTube API
        self.youtube_api = build('youtube', 'v3', developerKey=self.get_api_key())
        
        # Historical data storage
        self.history_db = pd.DataFrame(columns=['date', 'video_id', 'title', 'risk_score', 'violations'])
        self.load_history()
        
    def get_api_key(self):
        """Get YouTube API key from environment or file"""
        if os.getenv('YOUTUBE_API_KEY'):
            return os.getenv('YOUTUBE_API_KEY')
        try:
            with open('api_key.txt', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return 'YOUR_API_KEY'  # Replace with your actual API key

    def load_violation_categories(self):
        """Load violation categories with keywords and context rules"""
        return {
            "Hate Speech": {
                "keywords": ["hate", "racist", "sexist", "bigot", "nazi", "supremacist"],
                "context_rules": {
                    "positive": ["against", "fight against", "stand against", "oppose"],
                    "negative": ["promote", "support", "join", "believe in"]
                },
                "severity": 3
            },
            "Violence": {
                "keywords": ["kill", "murder", "assault", "shoot", "bomb", "terrorist"],
                "context_rules": {
                    "positive": ["condemn", "against", "prevent", "stop"],
                    "negative": ["how to", "instructions", "make", "build"]
                },
                "severity": 3
            },
            "Harassment": {
                "keywords": ["bully", "dox", "harass", "stalk", "revenge porn"],
                "context_rules": {
                    "positive": ["against", "report", "stop"],
                    "negative": ["how to", "expose", "leak"]
                },
                "severity": 3
            },
            "Sexual Content": {
                "keywords": ["porn", "sex", "nude", "onlyfans", "hentai"],
                "context_rules": {
                    "positive": ["warning", "education", "awareness"],
                    "negative": ["watch", "download", "free"]
                },
                "severity": 3
            },
            "Misinformation": {
                "keywords": ["fake news", "conspiracy", "hoax", "anti-vax"],
                "context_rules": {
                    "positive": ["debunk", "fact check"],
                    "negative": ["truth", "exposed", "real story"]
                },
                "severity": 2
            },
            "Copyright Issues": {
                "keywords": ["free movie", "full album", "cracked software"],
                "context_rules": {
                    "positive": ["warning", "legal"],
                    "negative": ["download", "free access"]
                },
                "severity": 2
            }
        }
    
    def load_demonetization_triggers(self):
        """Load demonetization triggers from CSV file"""
        try:
            return pd.read_csv('demonetization_triggers.csv')['term'].tolist()
        except:
            return ["demonetized", "not monetized", "adpocalypse", "limited ads"]
    
    def load_examples_db(self):
        """Load examples of compliant and non-compliant content"""
        return {
            "Hate Speech": {
                "compliant": "How to combat hate speech in online communities",
                "non_compliant": "Why hate speech should be protected as free speech"
            },
            "Violence": {
                "compliant": "Documentary about gang violence prevention",
                "non_compliant": "How to make a bomb at home"
            },
            "Harassment": {
                "compliant": "How to deal with online harassment",
                "non_compliant": "Where to find someone's private information"
            },
            "Sexual Content": {
                "compliant": "Sex education for teenagers",
                "non_compliant": "Explicit adult content tutorial"
            },
            "Misinformation": {
                "compliant": "Debunking common conspiracy theories",
                "non_compliant": "The truth about vaccines causing autism"
            },
            "Copyright Issues": {
                "compliant": "How copyright law protects creators",
                "non_compliant": "Download latest movies for free"
            }
        }
    
    def load_history(self):
        """Load historical analysis data from file"""
        try:
            self.history_db = pd.read_csv('history.csv', parse_dates=['date'])
        except FileNotFoundError:
            self.history_db = pd.DataFrame(columns=['date', 'video_id', 'title', 'risk_score', 'violations'])
    
    def save_history(self):
        """Save historical analysis data to file"""
        self.history_db.to_csv('history.csv', index=False)
    
    def setup_gui(self):
        """Set up the graphical user interface"""
        self.root.title("YouTube Content Guidelines Analyzer")
        self.root.geometry("1200x800")
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.root)
        self.analysis_tab = ttk.Frame(self.tab_control)
        self.history_tab = ttk.Frame(self.tab_control)
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.analysis_tab, text='Content Analysis')
        self.tab_control.add(self.history_tab, text='Channel History')
        self.tab_control.add(self.settings_tab, text='Settings')
        self.tab_control.pack(expand=1, fill="both")
        
        # Analysis Tab
        ttk.Label(self.analysis_tab, text="Video Title:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.title_entry = ttk.Entry(self.analysis_tab, width=80)
        self.title_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.analysis_tab, text="Video Description:").grid(row=1, column=0, padx=5, pady=5, sticky='nw')
        self.desc_text = tk.Text(self.analysis_tab, width=80, height=10)
        self.desc_text.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.analysis_tab, text="Thumbnail Image:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.thumbnail_path = tk.StringVar()
        ttk.Entry(self.analysis_tab, textvariable=self.thumbnail_path, width=70).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.analysis_tab, text="Browse", command=self.browse_image).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Button(self.analysis_tab, text="Analyze", command=self.analyze_gui).grid(row=3, column=1, pady=10)
        
        self.result_text = tk.Text(self.analysis_tab, width=100, height=20, state='disabled')
        self.result_text.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
        
        # History Tab
        self.history_tree = ttk.Treeview(self.history_tab, columns=('date', 'video_id', 'title', 'score'), show='headings')
        self.history_tree.heading('date', text='Date')
        self.history_tree.heading('video_id', text='Video ID')
        self.history_tree.heading('title', text='Title')
        self.history_tree.heading('score', text='Risk Score')
        self.history_tree.pack(fill='both', expand=True)
        
        self.plot_frame = ttk.Frame(self.history_tab)
        self.plot_frame.pack(fill='both', expand=True)
        
        ttk.Button(self.history_tab, text="Refresh History", command=self.update_history).pack(pady=5)
        ttk.Button(self.history_tab, text="Clear History", command=self.clear_history).pack(pady=5)
        
        # Settings Tab
        ttk.Label(self.settings_tab, text="YouTube API Key:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.api_key_entry = ttk.Entry(self.settings_tab, width=50)
        self.api_key_entry.grid(row=0, column=1, padx=5, pady=5)
        self.api_key_entry.insert(0, self.get_api_key())
        
        ttk.Button(self.settings_tab, text="Save Settings", command=self.save_settings).grid(row=1, column=1, pady=10)
        
        self.update_history()
    
    def browse_image(self):
        """Open file dialog to select thumbnail image"""
        filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        self.thumbnail_path.set(filename)
    
    def save_settings(self):
        """Save application settings"""
        api_key = self.api_key_entry.get()
        try:
            with open('api_key.txt', 'w') as f:
                f.write(api_key)
            self.youtube_api = build('youtube', 'v3', developerKey=api_key)
            messagebox.showinfo("Success", "Settings saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def clear_history(self):
        """Clear analysis history"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all history?"):
            self.history_db = pd.DataFrame(columns=['date', 'video_id', 'title', 'risk_score', 'violations'])
            self.save_history()
            self.update_history()
    
    def analyze_gui(self):
        """Handle analysis from GUI"""
        title = self.title_entry.get()
        description = self.desc_text.get("1.0", tk.END)
        thumbnail_path = self.thumbnail_path.get()
        
        if not title and not description:
            messagebox.showerror("Error", "Please enter title or description")
            return
        
        try:
            # Analyze text
            text_report = self.generate_report(title, description)
            
            # Analyze image if provided
            image_report = ""
            if thumbnail_path:
                image_report = self.analyze_image(thumbnail_path)
            
            # Display results
            self.result_text.config(state='normal')
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "=== TEXT ANALYSIS ===\n")
            self.result_text.insert(tk.END, text_report)
            
            if image_report:
                self.result_text.insert(tk.END, "\n\n=== IMAGE ANALYSIS ===\n")
                self.result_text.insert(tk.END, image_report)
            
            self.result_text.config(state='disabled')
            
            # Add to history
            risk_score = self.calculate_risk_score(text_report)
            violations = self.extract_violations(text_report)
            self.add_to_history(title, "", risk_score, violations)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def calculate_risk_score(self, report_text):
        """Calculate risk score from report text"""
        # Count the number of violation categories mentioned
        violation_count = sum(1 for category in self.violation_categories 
                             if f"[{category.upper()}]" in report_text)
        
        # Base score on violation count and severity
        base_score = min(100, violation_count * 20)
        
        # Adjust based on sentiment if available
        if "Sentiment:" in report_text:
            sentiment_line = [line for line in report_text.split("\n") if "Sentiment:" in line][0]
            if "Negative" in sentiment_line:
                base_score = min(100, base_score + 20)
            elif "Positive" in sentiment_line:
                base_score = max(0, base_score - 10)
        
        return base_score
    
    def extract_violations(self, report_text):
        """Extract violation categories from report text"""
        violations = []
        for category in self.violation_categories:
            if f"[{category.upper()}]" in report_text:
                violations.append(category)
        return ", ".join(violations)
    
    def init_ml_model(self):
        """Initialize the machine learning model"""
        try:
            model = load_model('youtube_guidelines_model.h5')
            # Load tokenizer
            with open('tokenizer_config.json', 'r') as f:
                tokenizer_config = f.read()
            self.tokenizer = tokenizer_from_json(tokenizer_config)
            return model
        except:
            # Create a simple model if no saved model exists
            model = Sequential()
            model.add(Embedding(10000, 128))
            model.add(LSTM(128))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
    
    def train_ml_model(self, texts, labels):
        """Train the machine learning model"""
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=200)
        
        self.ml_model.fit(data, np.array(labels), epochs=5, batch_size=32)
        self.ml_model.save('youtube_guidelines_model.h5')
        
        # Save tokenizer
        tokenizer_json = self.tokenizer.to_json()
        with open('tokenizer_config.json', 'w') as f:
            f.write(tokenizer_json)
    
    def predict_with_ml(self, text):
        """Make prediction using the ML model"""
        sequences = self.tokenizer.texts_to_sequences([text])
        data = pad_sequences(sequences, maxlen=200)
        return self.ml_model.predict(data)[0][0]
    
    def analyze_sentiment(self, text):
        """Perform sentiment analysis on text"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'assessment': 'Positive' if blob.sentiment.polarity > 0.1 else 
                         'Negative' if blob.sentiment.polarity < -0.1 else 'Neutral'
        }
    
    def analyze_image(self, image_path):
        """Analyze thumbnail image for policy violations"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return "Could not read image file"
            
            # Check for explicit content using simple computer vision
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            skin_pixels = self.detect_skin(img)
            skin_percentage = (np.sum(skin_pixels) / (img.shape[0] * img.shape[1])) * 100
            
            # OCR to extract text from image
            image_text = pytesseract.image_to_string(Image.open(image_path))
            text_violations = self.analyze_text(image_text, "")
            
            # Prepare report
            report = []
            report.append(f"Image Analysis Results:")
            report.append(f"- Skin percentage: {skin_percentage:.1f}% (high values may indicate nudity)")
            
            if text_violations:
                report.append("- Text in image contains potential violations:")
                for cat, words in text_violations.items():
                    report.append(f"  {cat}: {', '.join(set(words))}")
            
            return "\n".join(report)
        except Exception as e:
            return f"Image analysis error: {str(e)}"
    
    def detect_skin(self, image):
        """Simple skin detection algorithm"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower = np.array([0, 48, 80], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        
        # Threshold the HSV image
        skin_mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def add_to_history(self, title, video_id, risk_score, violations=""):
        """Add analysis result to history database"""
        new_entry = {
            'date': datetime.now(),
            'video_id': video_id,
            'title': title,
            'risk_score': risk_score,
            'violations': violations
        }
        self.history_db = pd.concat([self.history_db, pd.DataFrame([new_entry])], ignore_index=True)
        self.save_history()
        self.update_history()
    
    def update_history(self):
        """Update the history tab with latest data"""
        # Clear tree
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Add data (show most recent first)
        sorted_df = self.history_db.sort_values('date', ascending=False)
        for _, row in sorted_df.iterrows():
            self.history_tree.insert('', 'end', values=(
                row['date'].strftime("%Y-%m-%d %H:%M"),
                row['video_id'],
                row['title'][:50] + '...' if len(row['title']) > 50 else row['title'],
                row['risk_score']
            ))
        
        # Update plot
        self.plot_history()
    
    def plot_history(self):
        """Plot historical risk scores"""
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if len(self.history_db) < 2:
            return
        
        # Create new plot
        fig, ax = plt.subplots(figsize=(8, 4))
        self.history_db.sort_values('date', inplace=True)
        
        ax.plot(self.history_db['date'], self.history_db['risk_score'], marker='o')
        ax.set_title('Risk Score Trend Over Time')
        ax.set_ylabel('Risk Score')
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def analyze_text(self, title, description):
        """Analyze title and description for potential violations"""
        combined_text = f"{title} {description}"
        results = defaultdict(list)
        
        # Check for direct keyword matches
        for category, data in self.violation_categories.items():
            pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, data['keywords'])) + r')\b', re.IGNORECASE)
            matches = pattern.findall(combined_text)
            if matches:
                results[category].extend(matches)
        
        # Check context using NLP
        for category, data in self.violation_categories.items():
            context_violations = self.analyze_context(combined_text, category)
            for word, context in context_violations:
                if context == "negative":
                    results[category].append(word)
        
        # Check for demonetization triggers
        demonetization_issues = self.check_demonetization_triggers(combined_text)
        if demonetization_issues:
            results["Demonetization Risk"] = demonetization_issues
        
        return dict(results)
    
    def generate_report(self, title, description):
        """Generate a detailed violation report"""
        violations = self.analyze_text(title, description)
        sentiment = self.analyze_sentiment(f"{title} {description}")
        ml_risk_score = self.predict_with_ml(f"{title} {description}") * 100
        
        if not violations:
            return "No potential policy violations detected. Content appears compliant with YouTube guidelines.\n" \
                  f"Sentiment: {sentiment['assessment']} (Polarity: {sentiment['polarity']:.2f}, Subjectivity: {sentiment['subjectivity']:.2f})\n" \
                  f"ML Risk Prediction: {ml_risk_score:.1f}%"
        
        report = []
        report.append("YouTube Content Guidelines Analysis Report")
        report.append("="*50)
        report.append(f"Title: {title}")
        report.append(f"Sentiment: {sentiment['assessment']} (Polarity: {sentiment['polarity']:.2f}, Subjectivity: {sentiment['subjectivity']:.2f})")
        report.append(f"ML Risk Prediction: {ml_risk_score:.1f}%")
        report.append("\nPotential Violations Found:")
        
        total_severity = 0
        for category, words in violations.items():
            severity = self.violation_categories.get(category, {}).get("severity", 1)
            total_severity += severity * len(words)
            
            report.append(f"\n[{category.upper()}] (Severity: {severity}/3)")
            report.append(f"Matching words/phrases: {', '.join(set(words))}")
            
            # Add educational explanation
            explanation = self.get_category_explanation(category)
            report.append(f"Explanation: {explanation}")
            
            # Add suggestions
            suggestions = self.get_suggestions(category)
            report.append(f"Suggestions: {suggestions}")
            
            # Add examples if available
            examples = self.get_examples(category)
            if examples:
                report.append(f"Examples:\n- Compliant: {examples['compliant']}\n- Non-compliant: {examples['non_compliant']}")
        
        # Calculate overall risk score (0-100)
        risk_score = min(100, total_severity * 5 + ml_risk_score / 2)
        risk_level = "Low" if risk_score < 30 else "Medium" if risk_score < 70 else "High"
        
        report.append("\n" + "="*50)
        report.append(f"Overall Risk Score: {risk_score:.1f}/100 ({risk_level} Risk)")
        report.append("Note: This is an automated analysis. Final moderation decisions are made by YouTube.")
        
        return "\n".join(report)
    
    def get_category_explanation(self, category):
        """Return educational explanation for why content might be problematic"""
        explanations = {
            "Hate Speech": "Content that promotes hatred against groups based on race, religion, gender, or other protected attributes violates YouTube's policies and can lead to strikes or channel termination.",
            "Violence": "Graphic violence, instructions to commit violence, or content that glorifies violence may be age-restricted or removed. Content promoting self-harm is strictly prohibited.",
            "Harassment": "Targeted harassment, cyberbullying, or revealing private information (doxxing) can result in content removal and account penalties.",
            "Sexual Content": "Explicit sexual content, nudity, or content that sexualizes minors violates YouTube's policies and may be reported to authorities in extreme cases.",
            "Misinformation": "Content that spreads harmful misinformation, especially regarding medical treatments, elections, or major violent events, may be removed or demonetized.",
            "Copyright Issues": "Unauthorized use of copyrighted material may lead to content removal, demonetization, or legal action by copyright holders.",
            "Demonetization Risk": "Certain topics and keywords may trigger YouTube's automated systems to limit or disable monetization on your video."
        }
        return explanations.get(category, "This type of content may violate YouTube's community guidelines.")
    
    def get_suggestions(self, category):
        """Return suggestions for addressing potential violations"""
        suggestions = {
            "Hate Speech": "Consider removing discriminatory language and focusing on constructive dialogue. Educate rather than attack.",
            "Violence": "If discussing violent topics, provide context and analysis rather than glorification. Add content warnings where appropriate.",
            "Harassment": "Focus on ideas rather than individuals. Remove any personal attacks or private information.",
            "Sexual Content": "Ensure content is appropriate for YouTube's audience. Consider using euphemisms or academic language when discussing sensitive topics.",
            "Misinformation": "Verify facts with credible sources before publishing. Clearly distinguish between opinion and fact.",
            "Copyright Issues": "Use original content or properly licensed material. Consider fair use guidelines if using copyrighted material for commentary/education.",
            "Demonetization Risk": "Review YouTube's advertiser-friendly guidelines. Consider alternative phrasing for sensitive topics."
        }
        return suggestions.get(category, "Review YouTube's Community Guidelines and consider revising your content.")
    
    def analyze_youtube_video(self, video_id):
        """Use YouTube API to analyze an existing video"""
        try:
            request = self.youtube_api.videos().list(
                part="snippet,contentDetails,status",
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                return "Video not found"
            
            video = response['items'][0]
            title = video['snippet']['title']
            description = video['snippet']['description']
            thumbnail_url = video['snippet']['thumbnails']['high']['url']
            
            # Download thumbnail for analysis
            thumbnail_path = self.download_thumbnail(thumbnail_url)
            
            report = self.generate_report(title, description)
            if thumbnail_path:
                image_report = self.analyze_image(thumbnail_path)
                report += "\n\n=== THUMBNAIL ANALYSIS ===\n" + image_report
                os.remove(thumbnail_path)  # Clean up downloaded thumbnail
            
            # Add to history
            risk_score = self.calculate_risk_score(report)
            violations = self.extract_violations(report)
            self.add_to_history(title, video_id, risk_score, violations)
            
            return report
        except Exception as e:
            return f"Error analyzing YouTube video: {str(e)}"
    
    def download_thumbnail(self, url):
        """Download video thumbnail for analysis"""
        try:
            import urllib.request
            thumbnail_path = "temp_thumbnail.jpg"
            urllib.request.urlretrieve(url, thumbnail_path)
            return thumbnail_path
        except:
            return None
    
    def translate_and_analyze(self, text, target_lang='en'):
        """Translate non-English text and analyze"""
        try:
            translation = self.translator.translate(text, dest=target_lang)
            return self.generate_report(translation.text, "")
        except Exception as e:
            return f"Translation and analysis failed: {str(e)}"

def main():
    analyzer = YouTubeGuidelinesAnalyzer()
    analyzer.root.mainloop()

if __name__ == "__main__":
    main()
