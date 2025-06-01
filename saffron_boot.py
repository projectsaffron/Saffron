# -*- coding: utf-8 -*-
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify

# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set up Flask App
app = Flask("Saffron")

# Model registry
models = {
    "mistral": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "tokenizer": None,
        "model": None
    },
    "starcoder2": {
        "id": "bigcode/starcoder2-7b",
        "tokenizer": None,
        "model": None
    }
}

# Initialize model
def load_model(model_key):
    model_id = models[model_key]["id"]
    print(f"🚀 Loading {model_key.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    models[model_key]["tokenizer"] = tokenizer
    models[model_key]["model"] = model

# Decide which model to use
def decide_model(prompt):
    if not config.get("enable_auto_switch", True):
        return config.get("active_model", "mistral")
    if any(word in prompt.lower() for word in ["code", "function", "python", "javascript"]):
        return "starcoder2"
    return "mistral"

@app.route("/generate", methods=["POST"])
def generate():
    user_prompt = request.json.get("prompt", "")
    selected_model_key = decide_model(user_prompt)

    model_data = models[selected_model_key]
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]

    inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=250)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({
        "model": selected_model_key,
        "output": response.strip()
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "disk_usage": os.popen("df -h / | tail -1").read(),
        "gpu_status": os.popen("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv").read(),
        "status": "OK"
    })

if __name__ == "__main__":
    # Load required models based on config
    active_key = config.get("active_model", "mistral")
    fallback_key = config.get("fallback_model", "starcoder2")
    load_model(active_key)
    if config.get("enable_auto_switch", True):
        load_model(fallback_key)
app.run(host="0.0.0.0", port=7860)
