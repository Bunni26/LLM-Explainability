from models.groq_helper import llama_infer

try:
    result = llama_infer("Explain the difference between supervised and unsupervised learning.")
    print("✅ Groq Response:\n", result)
except Exception as e:
    print("❌ Error:", e)