from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .engine.response import get_emotionn
import os
from keras.models import model_from_json
# Create your views here.

@api_view(["POST"])
def bot_response(request):
    try:
        alamat = os.path.dirname(__file__)
        json_file = open(alamat + "/engine/model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(alamat + "/engine/model.h5")
        text = request.data['text']
        res = get_emotionn(loaded_model,text)
        
        return JsonResponse(res, safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
