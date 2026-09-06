import os
import sys
import time
import base64
import requests
from datetime import datetime

# Import configuration from ptz_shared.py
try:
    from ptz_shared import (
        BACKEND_BASE_URL,
        BACKEND_SERVER_URL,
        BACKEND_LOGIN_URL,
        BACKEND_REFRESH_URL,
        BACKEND_CAMERA_ID,
        BACKEND_SITE_ID,
        BACKEND_USERNAME,
        BACKEND_PASSWORD,
        BACKEND_AUTH_TOKEN,
        BACKEND_REFRESH_TOKEN,
        BACKEND_REQUEST_TIMEOUT,
    )
except ImportError:
    BACKEND_BASE_URL = "http://siteaense.kct.ac.in:8000"
    BACKEND_SERVER_URL = f"{BACKEND_BASE_URL}/api/ai-alerts/"
    BACKEND_LOGIN_URL = f"{BACKEND_BASE_URL}/api/auth/login/"
    BACKEND_REFRESH_URL = f"{BACKEND_BASE_URL}/api/auth/token/refresh/"
    BACKEND_CAMERA_ID = 1
    BACKEND_SITE_ID = 1
    BACKEND_USERNAME = "admin"
    BACKEND_PASSWORD = "Admin@123"
    BACKEND_AUTH_TOKEN = ""
    BACKEND_REFRESH_TOKEN = ""
    BACKEND_REQUEST_TIMEOUT = 5

def generate_dummy_b64_image():
    """Generates a tiny valid 1x1 JPEG base64 string for testing."""
    jpeg_bytes = bytes.fromhex(
        "ffd8ffe000104a46494600010101004000400000ffdb00430008060607060508"
        "0707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720"
        "222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001"
        "000101011100ffc4001f00000105010101010101000000000000000001020304"
        "05060708090a0bffda0008010100003f0037ffd9"
    )
    b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def perform_login():
    """Logs in to backend and returns fresh JWT access token."""
    print(f"\n[Auth] Logging in as '{BACKEND_USERNAME}' at {BACKEND_LOGIN_URL}...")
    try:
        r = requests.post(
            BACKEND_LOGIN_URL,
            json={"username": BACKEND_USERNAME, "password": BACKEND_PASSWORD},
            timeout=BACKEND_REQUEST_TIMEOUT,
        )
        if r.status_code in (200, 201):
            data = r.json()
            tokens_obj = (data.get("data") or {}).get("tokens") or {}
            token = (
                data.get("access")
                or data.get("access_token")
                or data.get("token")
                or (data.get("data") or {}).get("access")
                or tokens_obj.get("access")
            )
            if token:
                print("-> Login successful! Retrieved fresh JWT access token.")
                return token
            print(f"-> Login response OK but no access token found: {data}")
        else:
            print(f"-> Login failed: HTTP {r.status_code} - {r.text}")
    except Exception as e:
        print(f"-> Login request error: {e}")
    return ""

def test_delivery():
    print("=" * 60)
    print(" PPE ALERT BACKEND DELIVERY PROOF TESTER ")
    print("=" * 60)
    print(f"Target Backend URL: {BACKEND_SERVER_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 60)

    # 1. Connectivity test
    print("\n[Step 1] Testing basic TCP / HTTP connectivity...")
    try:
        ping_resp = requests.get(BACKEND_SERVER_URL, timeout=BACKEND_REQUEST_TIMEOUT)
        print(f"-> Ping Success! Server responded with HTTP {ping_resp.status_code}")
        print(f"-> Server Header: {ping_resp.headers.get('Server', 'Unknown')}")
    except Exception as e:
        print(f"-> Connection Error: {e}")
        print("RESULT: Cannot reach backend server. Check IP, Port, Firewall or VPN.")
        return

    # 2. Authentication check - perform fresh login
    token = perform_login()

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print(f"-> Using Auth Header: Bearer {token[:20]}...")
    else:
        print("-> WARNING: No auth token obtained. Request will fail with HTTP 401.")

    # 3. Send test alert payload
    print("\n[Step 3] Sending test PPE Alert payload to backend...")
    b64_img = generate_dummy_b64_image()
    
    payload = {
        "type": "helmet_violation",
        "severity": "CRITICAL",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "OPEN",
        "camera": int(BACKEND_CAMERA_ID),
        "site": int(BACKEND_SITE_ID),
        "snapshot": b64_img,
    }

    try:
        start_t = time.time()
        resp = requests.post(
            BACKEND_SERVER_URL,
            json=payload,
            headers=headers,
            timeout=BACKEND_REQUEST_TIMEOUT,
        )
        elapsed = time.time() - start_t

        print("-" * 60)
        print(" PROOF OF RECEIPT / BACKEND RESPONSE ")
        print("-" * 60)
        print(f"HTTP Status Code : {resp.status_code}")
        print(f"Roundtrip Time   : {elapsed:.3f} seconds")
        print(f"Server           : {resp.headers.get('Server', 'N/A')}")
        print(f"Date Header      : {resp.headers.get('Date', 'N/A')}")
        print(f"Response Body    : {resp.text}")
        print("-" * 60)

        proof_file = "alert_delivery_proof.txt"
        with open(proof_file, "w") as pf:
            pf.write(f"Timestamp: {datetime.now().isoformat()}\n")
            pf.write(f"Target URL: {BACKEND_SERVER_URL}\n")
            pf.write(f"HTTP Status: {resp.status_code}\n")
            pf.write(f"Response Headers:\n{dict(resp.headers)}\n")
            pf.write(f"Response Body:\n{resp.text}\n")

        print(f"\nProof log saved to '{proof_file}'")

        if resp.status_code in (200, 201):
            print("\n✅ PROOF VERIFIED: Backend successfully RECEIVED and CREATED the alert (HTTP 201 Created).")
            print("Give 'alert_delivery_proof.txt' or the returned alert_id to your backend team.")
        elif resp.status_code in (400, 422):
            print("\n⚠️ PACKET REACHED BACKEND: Backend received request but rejected validation (HTTP 400/422).")
            print("This proves network connectivity is OK, but payload structure needs alignment.")
        elif resp.status_code == 401:
            print("\n⚠️ PACKET REACHED BACKEND: Backend received request but rejected authentication (HTTP 401 Unauthorized).")
            print("This proves network connectivity is OK, but valid JWT auth token/credentials are required.")
        else:
            print(f"\nBackend returned status code {resp.status_code}.")

    except Exception as e:
        print(f"-> Failed to send alert request: {e}")

if __name__ == "__main__":
    test_delivery()
