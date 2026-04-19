# setup.py
# Run this once after cloning: python setup.py
# It preprocesses data and builds the ML model from scratch.

import os
import sys

print("=" * 55)
print("  Research Paper Dashboard — Setup Script")
print("=" * 55)

# Step 1: Check data file exists
if not os.path.exists("data/papers.csv"):
    print("\n❌ ERROR: 'data/papers.csv' not found.")
    print("   Please add your dataset to the data/ folder first.")
    sys.exit(1)

# Step 2: Preprocess
print("\n📦 Step 1/2 — Preprocessing data...")
os.system("python preprocess.py")

# Step 3: Build model
print("\n🤖 Step 2/2 — Building TF-IDF model...")
os.system("python recommender.py")

print("\n" + "=" * 55)
print("✅ Setup complete! Launch the app with:")
print("   streamlit run app.py")
print("=" * 55)