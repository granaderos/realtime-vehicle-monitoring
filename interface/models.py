from django.db import models

# Create your models here.
class PlateNumberImage(models.Model):
    image = models.FileField(upload_to="images/", default="")
    name = models.CharField(max_length=64, default="")