from django import forms
from django.conf import settings


class PredictionForm(forms.Form):
    nom = forms.CharField(
        max_length=100, label="Nom",
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )
    age = forms.IntegerField(
        label="Âge", min_value=0, max_value=120,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    file = forms.ImageField(
        label="Téléchargez une radiographie",
        widget=forms.FileInput(attrs={"class": "form-control"}),
    )

    def clean_file(self):
        file = self.cleaned_data["file"]
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file.size > max_bytes:
            raise forms.ValidationError(
                f"Fichier trop volumineux (maximum {settings.MAX_UPLOAD_SIZE_MB} Mo)."
            )
        return file
