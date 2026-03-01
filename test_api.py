"""
Test script for Khmer Gender Classification API
"""
import requests
import json


def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"

    print("🇰🇭 Testing Khmer Gender Classification API")
    print("=" * 70)

    # Test 1: Health check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Single prediction
    print("\n2. Testing Single Prediction...")
    test_names = ["ចន្ទា", "សុខ", "រដ្ឋា", "ហ្វុង"]

    for name in test_names:
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"name": name}
            )
            result = response.json()
            print(f"\n   Name: {result['name']}")
            print(f"   KCCs: {' + '.join(result['kccs'])}")
            print(f"   Gender: {result['gender']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Probability: {result['probability']}")
        except Exception as e:
            print(f"   ✗ Error for '{name}': {e}")

    # Test 3: Batch prediction
    print("\n3. Testing Batch Prediction...")
    try:
        batch_names = ["ចន្ទា", "សុខ", "រដ្ឋា", "និសា", "ហេង"]
        response = requests.post(
            f"{base_url}/batch_predict",
            json={"names": batch_names}
        )
        result = response.json()
        print(f"   Total predictions: {result['count']}")

        for pred in result['predictions']:
            print(f"\n   {pred['name']} → {pred['gender']} ({pred['confidence']}%)")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("✓ Testing complete!")


if __name__ == "__main__":
    test_api()
