from django.apps import AppConfig


class PredictionsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictions"
    # Le modèle est chargé à la première prédiction (voir predictions/ml.py)
    # pour que `manage.py migrate`, `test`, etc. restent rapides.
