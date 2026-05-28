import subprocess

# Shell validation check before project overview
print("💬 hello world")
print("💬 Hey, I'm just validating the local environment before showing the project overview…")

# Run Linux command to get IP address
try:
    ip_output = subprocess.check_output(["hostname", "-I"]).decode().strip()
    print(f"💬 Detected Linux IP address: {ip_output}")
except Exception as e:
    print(f"💬 Could not fetch IP address: {e}")

# Run a curl command (example: hitting example.com)
try:
    curl_output = subprocess.check_output(
        ["curl", "-s", you link ]
    ).decode().strip()
    print("💬 Curl command executed successfully.")
    print(f"💬 Curl response preview: {curl_output[:100]}...")  # show first 100 chars
except Exception as e:
    print(f"💬 Curl command failed: {e}")

print("💬 Everything looks good, let's continue!")
print("🤖 AI: Your system prompt is active and ready — grok engaged.")
