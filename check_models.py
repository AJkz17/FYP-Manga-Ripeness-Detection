import google.generativeai as genai

# PASTE YOUR ACTUAL KEY HERE
genai.configure(api_key="AIzaSyBBadNa9WeCT3PdZ6vd-4L4wVfiFAsiYHY")

print("Checking available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")
    