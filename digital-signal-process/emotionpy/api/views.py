from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt

from Model import Model
from speechpy import preprocessor
from speechpy.raw_socket import TCPSocket

import threading

model = Model()
model.load_dataset()
model.train()

# Create your views here.
def index(request):
    return HttpResponse('Hello Django!')

@csrf_exempt
def predict(request):

    if request.method == 'GET':
        return JsonResponse({
            "message": "Only POST method is available."
        })
    
    elif request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        # data = body['data']
        url = body['url']

        sockett = TCPSocket()
        SEOWJIN = '192.168.166.153'
        ADDRESS = '192.168.162.195'
        sockett.connect(ADDRESS, 5000)
        print('url:', url)
        sockett.send(url + '\n')
        gscore = sockett.receive()
        sockett.close()

        data = preprocessor.download_and_process(url)
        prediction = model.predict(data)
        return JsonResponse({
            "message": "Hello Django!",
            "predictions": prediction
        })