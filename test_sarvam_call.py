"""
First Working Call to Sarvam AI (Chat Completions / Classical Tamil Interpretation).
Reads API key securely from environment; never prints or hardcodes the secret key.
"""

import os
import sys

# Load .env file securely into os.environ if present
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

api_key = os.environ.get("SARVAM_API_KEY")
if not api_key:
    print("Error: SARVAM_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

from sarvamai import SarvamAI

# Initialize Sarvam AI client using the environment variable
client = SarvamAI(api_subscription_key=api_key)

print("Connecting to Sarvam AI (Model: sarvam-105b)...")

# Classical Tamil epigraph from Mangulam (3rd c. BCE Pandya inscription)
test_inscription = "கணிய் நந்தஆஸிரிய் இகுவன் கட்அளன் வழுதிஇ கொடிபித பாளிய்"

response = client.chat.completions(
    model="sarvam-105b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert Classical Tamil linguistic and epigraphical scholar. "
                "Interpret historical Tamil stone inscriptions into clear modern Tamil and English. "
                "Preserve historical names and epigraphical terms."
            )
        },
        {
            "role": "user",
            "content": f"Interpret this ancient Tamil inscription and explain its meaning:\n\n{test_inscription}"
        }
    ],
    reasoning_effort=None,  # Disables thinking tokens so output is returned directly in content
    max_tokens=1000
)

choice = response.choices[0]
content = choice.message.content
if not content and hasattr(choice.message, "reasoning_content"):
    content = choice.message.reasoning_content

print("\n--- Sarvam AI Response ---")
print(content)
print("\n--- Execution Successful ---")

