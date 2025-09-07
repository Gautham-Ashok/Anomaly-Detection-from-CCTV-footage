import requests
import json


def test_api():
    base_url = "http://localhost:5000"

    print("Testing API endpoints...")

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")

    # Test categories endpoint
    try:
        response = requests.get(f"{base_url}/categories")
        print(f"Categories: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Categories failed: {e}")

    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/stats")
        print(f"Stats: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Stats failed: {e}")


if __name__ == "__main__":
    test_api()