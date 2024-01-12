import requests

# Replace 'YOUR_API_ENDPOINT' with the actual API endpoint you want to use
api_endpoint = 'https://threed-model-management.onrender.com/model-management-system/internal/ml/classification/store'

# Make a GET request to the API
response = requests.get(api_endpoint)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the response as JSON (assuming the API returns JSON data)
    data = response.json()

    # Now you can work with the data
    print(data)
else:
    # If the request was not successful, print an error message
    print(f"Error: {response.status_code}")
    print(response.text)