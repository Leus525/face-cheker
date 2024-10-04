from django.shortcuts import render
from django.http import HttpResponse
#from .forms import UploadFileForm
from django.views.generic import ListView
from .models import Faces
from insightface.app import FaceAnalysis
import cv2
import os
#from upload import handle_uploaded_file
from django.http import HttpResponseRedirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from pgvector.django import L2Distance
import numpy as np

# Create your views here.

class FaceSite(ListView):
    model = Faces

def show(request):

    return render(request, "faces/index.html")

def new(request):
     if request.method == 'POST' and request.FILES['photo']:

         myfile = request.FILES.getlist('photo')
         mid_faces=[]
         for file in myfile:
             fs = FileSystemStorage()
             filename = fs.save(file.name, file)
             uploaded_file_url = fs.url(filename)
             path_url = '/home/leus/mediaFoto/' + filename
             print(uploaded_file_url)
             app = FaceAnalysis(name="buffalo_l",providers="CUDAExecutionProvider")
             app.prepare(ctx_id=0, det_size=(256, 256))  #подготовка нейросети
             img = cv2.imread(path_url) #считываем изображение
             print(file)
             print("-----------------------------------------")
             print(" ===== ")
             faces = app.get(img) #ищем лица на изображении и получаем информацию о них
             face=0

             for face in faces:
                if face != 0:
                    mid_faces.append(face.embedding)

         Faces.objects.create(embedding=np.mean(mid_faces, axis=0), name=myfile[0])
         return render(request, 'faces/index.html')
     return render(request, 'faces/index.html')
def compare(request):
     accuracy = 23
     if not os.path.exists("report " + str(accuracy)):
        os.mkdir("report " + str(accuracy))
     if request.method == 'POST' and request.FILES['photo']:
         cv2.destroyAllWindows()
         print("-----------------------------------------")
         face=0
         myfile = request.FILES.getlist('photo')
         loaded=len(myfile)
         checked=0
         name_file = "report " + str(accuracy) + "/" + str(myfile[0]) + ".txt"
         f = open(name_file, "w")
         unfounded=[]
         belongs=[]
         belongs_clear=[]
         folders=[]
         unrecognized=[]
         # get list of uploaded files
         for file in myfile:
             fs = FileSystemStorage()
             filename = fs.save(file.name, file)
             uploaded_file_url = fs.url(filename)
             path_url = '/home/leus/mediaFoto/' + filename
             app = FaceAnalysis(name="buffalo_l",providers="CUDAExecutionProvider")
             app.prepare(ctx_id=0, det_size=(256, 256))  #подготовка нейросети
             img = cv2.imread(path_url) #считываем изображение
             print("-----------------------------------------")
             faces = app.get(img) #ищем лица на изображении и получаем информацию о них
             #check every face on img
             unrecognized.append(filename)

             for face in faces:

                 if face != 0:
                    # showing implemented face in window
                     x, y, x2, y2 = face.bbox #получаем границы лица
                     cropped = img[int(y):int(y2), int(x):int(x2)] #вырезаем лицо из изображения
                     if cropped.shape[1]:
                        cv2.imshow('image', cropped) #показываем лицо
                        cv2.waitKey(0)
                     else:
                        unfounded.append(filename)
                     #cv2.destroyAllWindows()
                      # check if implemented face is already in database
                     fbase = Faces.objects.alias(distance=L2Distance('embedding', face.embedding)).filter(distance__lt=accuracy)
                     bas = 0
                     for bas in fbase:
                          #if face is found then it gonna show on screen
                         if bas != 0:
                             path_url = '/home/leus/mediaFoto/' + bas.name
                             app = FaceAnalysis(name="buffalo_l", providers="CUDAExecutionProvider")
                             app.prepare(ctx_id=0, det_size=(256, 256))  #подготовка нейросети
                             img = cv2.imread(path_url) #считываем изображение
                             new_face = app.get(img)

                             # !!!! check which one face should be displayed
                             #print(new_face)
                             for face_one in new_face:
                               #  print(face_one)
                                if np.linalg.norm(face_one.embedding - face.embedding) <= accuracy:
                                     x, y, x2, y2 = face_one.bbox #получаем границы лица
                                     cropped = img[int(y):int(y2), int(x):int(x2)] #вырезаем лицо из изображения
                                     cv2.imshow('image', cropped) #показываем лицо
                                     cv2.waitKey(0)
                                     checked+=1
                                     var =  str(filename) + " - " +  str(bas.folder)
                                     folders.append(bas.folder)
                                     belongs.append(var)
                                     belongs_clear.append(filename)
                             print("=============--=============")
                 else:
                     print("no face")
             cv2.destroyAllWindows()
             print(checked, "/", loaded)
             print(100*checked/loaded, "%")


         new_folder = np.array(folders)
         vals, counts=np.unique(new_folder, return_counts=True)
         index = np.argmax(counts)
         number = "- " + str(vals[index])
         print("----Another result-----")
         fol = []
         for i in belongs:
             if number in i:
                 fol.append(i)
         checked2 = len(set(fol))
         #print(set(fol))
         print(checked2, "/", loaded)
         print(100*checked2/loaded, "%")
         #load statistics
         unrecognized_new = [i for i in unrecognized if i not in belongs_clear]
         print(unrecognized_new)
         f.write("-----------------Results----------------------\n")
         f.write(str(checked) + "<- faces found  " +  str(checked2) + "/" + str(loaded) + "\n")
         f.write(str(100*checked2/loaded) + "%" + "\n")
         print("-----------------Unfound photos----------------------")
         f.write("-----------------Unfound photos----------------------\n")
         for face in unfounded:
             print(face)
             f.write(face + "\n")
             print("-------------------------------------------------")
         print("")
         f.write("-----------------Photo belongs to----------------------\n")
         print("-----------------Photo belongs to----------------------")
         for face in belongs:
             print(face)
             f.write(face + "\n")
             print("-------------------------------------------------")
         print("")
         f.write("-----------------Not recognized faces----------------------\n")
         print("-----------------Not recognized faces----------------------")
         for face in unrecognized_new:
             print(face)
             f.write(face + "\n")
             print("-------------------------------------------------")
         f.close()
         return render(request, 'faces/index.html')
     return render(request, 'faces/index.html')

# !!!! В базу данный записываем еще имя фотографии и потом находим ее в локальной папке и выводим ее на экран


