# test_extractor.py
import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Test 1: Check API key exists
print("API Key found:", bool(os.environ.get("GROQ_API_KEY")))
print("API Key starts with:", os.environ.get("GROQ_API_KEY", "")[:8])

# Test 2: Simple Groq call
print("\nTesting basic Groq call...")
try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": "Say hello in JSON: {\"message\": \"hello\"}"}
        ],
        temperature=0.0,
        max_tokens=50,
    )
    print("SUCCESS:", response.choices[0].message.content)
except Exception as e:
    print("FAILED:", str(e))

# Test 3: Test extraction prompt directly
print("\nTesting extraction prompt...")
abstract = """
We present a novel deep learning approach for image classification
using convolutional neural networks. Our method achieves 95% accuracy
on ImageNet. The main limitation is high computational cost.
Future work includes applying this to video understanding.
"""

prompt = """Extract from this abstract and return ONLY valid JSON with these exact keys:
{"problem": "...", "methodology": "...", "contributions": "...", "results": "...", "limitations": "...", "future_work": "..."}

Abstract: """ + abstract

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. No markdown. No explanation."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content
    print("RAW RESPONSE:")
    print(repr(raw))
    print("\nFormatted:")
    print(raw)
except Exception as e:
    print("FAILED:", str(e))