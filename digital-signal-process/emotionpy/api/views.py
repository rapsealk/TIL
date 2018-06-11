from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt

from Modelnlp import Model
# from Model import Model
from speechpy import preprocessor
from speechpy.raw_socket import TCPSocket

model = Model()
#model.load_dataset()
model.load_dataset_for_gapi(external=False)
model.train()
#model.load_dataset()
#model.train()

# Create your views here.
def index(request):
    return HttpResponse('Hello Django!')

@csrf_exempt
def predict(request):

    if request.method == 'GET':

        url = 'https://firebasestorage.googleapis.com/v0/b/kaubrain418.appspot.com/o/lsj%2F%5B%E1%84%8B%E1%85%A7%E1%86%AB%E1%84%87%E1%85%A1%E1%86%AF%5D%20Angry18.WAV?alt=media&token=9f2aee11-8b35-49e6-b05b-7b64700b7bd4'
        data = preprocessor.download_and_process(url)
        prediction = model.predict(data, float(0))

        return JsonResponse({
            "message": "Only POST method is available.",
            "predictions": prediction
        })
    
    elif request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        # data = body['data']
        url = body['url']

        sockett = TCPSocket()
        gapi = sockett.read_gapi(url)

        data = preprocessor.download_and_process(url)
        prediction = model.predict(data, gapi)

        return JsonResponse({
            "message": "Hello Django!",
            "predictions": prediction
        })