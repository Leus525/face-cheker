from django.db import models
from pgvector.django import VectorField
#
#
class Faces(models.Model):
         id = models.AutoField(primary_key=True)
         embedding = VectorField(dimensions=512, null=True)
         name = models.CharField(max_length=400)
         folder = models.IntegerField()
# class Phot(models.Model):
#        name = models.CharField(max_length=400)

