AI-Powered UX Analytics
- An AI-powered UX analytics and conversion optimization system that simulates realistic user clickstream behavior, predicts the next user action using a Bidirectional LSTM and generates UX recommendations using a LLM.
- The project is designed to help product teams understand how users navigate through a platform, where users drop off, what users are likely to do next and how UX can be improved to increase conversions.

Features:
- Synthetic clickstream dataset generation
- Multiple user personas and behavioral funnels
- Funnel stage tracking
- Navigation path simulation
- User journey visualization
- Session analysis and EDA
- Sliding-window sequence generation
- Context-aware sequence preprocessing
- BiLSTM next-click prediction
- Contextual embeddings for persona type, device type, traffic source and funnel stage
- Early stopping and learning rate scheduling
- Top K next click prediction
- LLM powered UX recommendation generation
- Automated UX recommendation report generation
- Confusion matrix and training visualization charts

Tech Stack:
- Python
- PyTorch
- NumPy
- scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Hugging Face Inference API
- Meta Llama 3.1 / 3.2 Instruct
- BiLSTM

Workflow:
1. Synthetic Clickstream Generation
2. Exploratory Data Analysis
3. Sequence Preprocessing
4. Sliding Window Sequence Creation
5. Feature Encoding & Padding
6. BiLSTM Training
7. Next Click Prediction
8. LLM Prompt Construction
9. UX Recommendation Generation
10. Final UX Recommendation Report Generation

Installation:
1. Clone the Repository: 
git clone <your-repo-link>
cd AI-UX-Project
2. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Create Hugging Face Token
Then run: <bash> export HF_TOKEN="your_huggingface_token"
5. Running the Project
python3 src/data_generation/generate_clickstream.py
python3 src/visualization/eda_clickstream
python3 src/preprocessing/preprocess_sequences.py
python3 src/modelling/train_lstm.py
python3 src/llm/generate_ux_recommendations.py

Sample Output:
Sample 1
────────────────────────────────────────────────────────────
Context : persona=Buyer  device=desktop  traffic=email  funnel=discovery
Journey : Home -> Search -> Home -> Search -> Home
Predicted next : Search (88.57%)
Ground truth   : Search

────────────────────────────────────────────────────────────────────────
  SESSION 1 — HOW TO CONVERT THIS USER
────────────────────────────────────────────────────────────────────────
1. WHY ARE THEY LEAVING?
- The predicted next click to "Product" indicates the user is looking for clarity on product details. This might be due to:
- Lack of clear call-out on product options or variations (e.g., color, size,or features).
- Insufficient information or demos for specific products on the current page.
- Unclear pricing or promotions that make them want to explore individual products.

2. WHAT TO ADD OR CHANGE ON THIS PAGE RIGHT NOW
To keep the user on the page, consider adding or changing:
- Product Variations**: A grid or tile layout highlighting key product options (e.g., color, size, or material) along with images or quick descriptions.
- Product Demos or Videos**: Embedded videos or carousel sections featuring product demos or tutorials to showcase key features.
- Product Information**: A highlighted section summarizing key product benefits, specifications, or FAQs to address any lingering questions.

3. CTA FIX
For a returning user arriving via email on a desktop device at the discovery stage, a specific CTA fix could be:
- CTA Copy: "Unlock Exclusive Offers & Benefits"
- CTA Placement:** Prominent button located below the main content, at eye level.
- The CTA is surrounded by a bright orange border, contrasting with white and gray UI elements, and a font weight of 500 to create a sense of urgency.

Future Recommendations:
- Real-Time Prediction Dashboard - deploy the model with Streamlit for real time UX insights.
- Replace synthetic clickstream data with real analytics datasets.
- Generate recommendations customized for each persona type.