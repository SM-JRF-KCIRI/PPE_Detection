import requests
from requests.auth import HTTPDigestAuth

CAMERA_IP = "192.168.1.126"
USERNAME = "admin"
PASSWORD = "Admin@123"

DEVICE_URL = f"http://{CAMERA_IP}:80/onvif/device_service"

soap = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
    <s:Body>
        <tds:SystemReboot/>
    </s:Body>
</s:Envelope>
"""

response = requests.post(
    DEVICE_URL,
    data=soap,
    headers={"Content-Type": "application/soap+xml"},
    auth=HTTPDigestAuth(USERNAME, PASSWORD),
    timeout=10
)

print("Status:", response.status_code)
print(response.text)
