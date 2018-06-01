from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt

# from Model import Model
from tfModel import Model

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
        """
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        data = body['data']
        prediction = model.predict(data)
        """
        return JsonResponse({
            "message": "Hello Django!",
            "predictions": [] # prediction
        })