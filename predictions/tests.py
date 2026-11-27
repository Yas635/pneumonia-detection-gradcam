from unittest import mock

from django.test import TestCase
from django.urls import reverse

from . import ml
from .models import Submission


class PagesTests(TestCase):
    def test_home_page_loads(self):
        response = self.client.get(reverse("predict_pneumonia"))
        self.assertEqual(response.status_code, 200)

    def test_dashboard_counts_all_submissions(self):
        for i in range(15):
            Submission.objects.create(
                nom=f"p{i}", age=30, prediction="Pneumonia" if i % 3 == 0 else "Normal"
            )
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_submissions"], 15)
        total_in_groups = sum(
            g["Pneumonia"] + g["Normal"] for g in response.context["age_groups"].values()
        )
        self.assertEqual(total_in_groups, 15)  # toutes les soumissions sont comptées


class PredictionTests(TestCase):
    def test_invalid_form_shows_error(self):
        response = self.client.post(reverse("predict_pneumonia"), {"nom": "", "age": ""})
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context["error"])

    def test_missing_model_gives_friendly_error(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        # PNG 1x1 valide
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
            b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe"
            b"\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        upload = SimpleUploadedFile("x.png", png, content_type="image/png")
        with mock.patch.object(ml, "get_model", side_effect=ml.ModelNotAvailable("absent")):
            response = self.client.post(
                reverse("predict_pneumonia"), {"nom": "Test", "age": 30, "file": upload}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("absent", response.context["error"])
        self.assertEqual(Submission.objects.count(), 0)
