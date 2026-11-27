from django.db import models


class Submission(models.Model):
    nom = models.CharField(max_length=100)
    age = models.PositiveIntegerField()
    prediction = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.nom} - {self.prediction}"
