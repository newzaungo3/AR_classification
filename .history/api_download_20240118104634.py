import requests
import cv2
import urllib.request 

api_endpoint = "https://threed-model-management.onrender.com/model-management-system/internal/ml/classification/store"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJCbnVIRzQxbXpkdE1ZMjhuM3JSeCIsIm9yZ2FuaXphdGlvbklkIjoieU9oOWhBYmQ0RmZ1OTRtUHBLcjgiLCJpYXQiOjE3MDEwNjE5NTl9.cZ2xkoMJzvyaMMUYYm15XiG4xA9YmFS8fuZJBpf6d4Y"
}
# Make a GET request to the API
response = requests.post(api_endpoint,headers=headers)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the response as JSON (assuming the API returns JSON data)
    data = response.json()

    # Now you can work with the data
    print(len(data))
    for i in range(len(data)):
        #print(i)
        #print(type(data[i]))
        print(data[i])
        print(type(data[i]['markers']))
        print(f"marker: {data[i]['markers']}")
        marker_list = data[i]['markers']
        content_list = data[i]['contents']
        print(content_list)
        for j in range(len(marker_list)):
            print(marker_list[j]['s3Url'])
            url = marker_list[j]['s3Url']
            urllib.request.urlretrieve(url, f"{marker_list[j]['s3s3FileName']}") 


        # Accessing the content of each dictionary
else:
    # If the request was not successful, print an error message
    print(f"Error: {response.status_code}")
    print(response.text)