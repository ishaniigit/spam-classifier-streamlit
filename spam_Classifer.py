# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 2: Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only useful columns
df.columns = ['label', 'message']  # Rename for clarity

# Optional: See first few rows
print(df.head())

# Step 3: Preprocess labels (spam/ham → 1/0)
le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])

# Step 4: Convert message text to vectors
cv = CountVectorizer()
X = cv.fit_transform(df['message'])  # Features
y = df['label_num']  # Labels

# Check shape of data
print("Feature matrix shape:", X.shape)
# Step 5: Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Step 6: Train the model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Create the model
model = MultinomialNB()

# Train the model with training data
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Step 7: Predict custom message
sample = ["Congratulations! You've won ₹1,00,000! Claim now by clicking this link."]
sample_vec = cv.transform(sample)
predicted = model.predict(sample_vec)

# Show result
print("Prediction:", le.inverse_transform(predicted))
test_messages = [
    "Hey! Are we still meeting at 6?",
    "URGENT: Your account has been compromised! Click the link to secure it.",
    "You have been selected for a cash prize worth ₹50,000!",
    "Let's catch up this weekend. Miss talking to you!",
    "Exclusive deal just for you!!! Buy now and get 80% off!",
    "Your Amazon order has been shipped and will arrive tomorrow.",
    "Call me when you're free.",
    "Win a free vacation to Dubai. Just answer this simple quiz!"
]

test_vectors = cv.transform(test_messages)
predictions = model.predict(test_vectors)

for message, label in zip(test_messages, le.inverse_transform(predictions)):
    print(f"\n📨 Message: {message}")
    print(f"🔍 Prediction: {label}")
    import joblib

joblib.dump(model, 'spam_model.pkl')
joblib.dump(cv, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')