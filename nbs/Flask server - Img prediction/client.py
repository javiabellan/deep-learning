import requests
resp = requests.post("http://localhost:5000/predict",
                     files={"my_img_file": open('cardigan.jpg','rb')})
print(resp.json())