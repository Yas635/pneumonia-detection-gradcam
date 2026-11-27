import base64
import io
import uuid
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import render
from django.template.loader import get_template
from xhtml2pdf import pisa

from . import ml
from .forms import PredictionForm
from .models import Submission

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _age_group(age: int) -> str:
    if age <= 20:
        return "0-20"
    if age <= 40:
        return "21-40"
    if age <= 60:
        return "41-60"
    return "60+"


def predict_pneumonia(request):
    error = None
    prediction = None

    if request.method == "POST":
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            nom = form.cleaned_data["nom"]
            age = form.cleaned_data["age"]
            file = form.cleaned_data["file"]

            try:
                raw = file.read()
                img = ml.preprocess_image(raw)
                model = ml.get_model()
                prediction = ml.predict(model, img)

                # Dossier unique par requête : deux utilisateurs simultanés
                # n'écrasent plus les images de l'autre.
                request_id = uuid.uuid4().hex
                ext = Path(file.name).suffix.lower()
                ext = ext if ext in ALLOWED_EXTENSIONS else ".png"
                upload_rel = Path("uploads") / f"{request_id}{ext}"
                upload_path = Path(settings.MEDIA_ROOT) / upload_rel
                upload_path.parent.mkdir(parents=True, exist_ok=True)
                upload_path.write_bytes(raw)

                output_dir = Path(settings.MEDIA_ROOT) / "generated_images" / request_id
                ml.generate_grad_cam(model, img, output_dir)

                submission = Submission.objects.create(nom=nom, age=age, prediction=prediction)

                media = settings.MEDIA_URL
                context = {
                    "nom": nom,
                    "age": age,
                    "prediction": prediction,
                    "image_path": request.build_absolute_uri(f"{media}{upload_rel.as_posix()}"),
                    "grad_cam_url": request.build_absolute_uri(
                        f"{media}generated_images/{request_id}/grad_cam_image.png"
                    ),
                    "overlay_url": request.build_absolute_uri(
                        f"{media}generated_images/{request_id}/overlay_image.png"
                    ),
                    "prediction_date": submission.created_at,
                }
                html = get_template("predictions/prediction_pdf.html").render(context)
                pdf_buffer = BytesIO()
                status = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), dest=pdf_buffer)
                if status.err:
                    error = "Erreur lors de la génération du PDF."
                else:
                    pdf_buffer.seek(0)
                    return FileResponse(
                        pdf_buffer, as_attachment=True, filename="rapport_prediction.pdf"
                    )
            except ml.ModelNotAvailable as exc:
                error = str(exc)
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                error = f"Une erreur est survenue : {exc}"
        else:
            error = "Le formulaire est invalide : " + " ".join(
                msg for errs in form.errors.values() for msg in errs
            )
    else:
        form = PredictionForm()

    return render(
        request,
        "predictions/index.html",
        {"form": form, "prediction": prediction, "error": error},
    )


def dashboard(request):
    submissions_qs = Submission.objects.all()
    total_submissions = submissions_qs.count()
    pneumonia_count = submissions_qs.filter(prediction="Pneumonia").count()
    normal_count = total_submissions - pneumonia_count
    pneumonia_percentage = (
        pneumonia_count / total_submissions * 100 if total_submissions else 0
    )

    # Répartition par tranche d'âge sur TOUTES les soumissions
    # (auparavant, seules les 10 dernières étaient comptées).
    age_groups = defaultdict(lambda: {"Pneumonia": 0, "Normal": 0})
    for age, prediction in submissions_qs.values_list("age", "prediction"):
        label = "Pneumonia" if prediction == "Pneumonia" else "Normal"
        age_groups[_age_group(age)][label] += 1

    latest = submissions_qs.order_by("-created_at")[:10]

    sizes = [pneumonia_count, normal_count] if total_submissions else [1, 0]
    fig, ax = plt.subplots()
    ax.pie(
        sizes, labels=["Pneumonia", "Normal"], autopct="%1.1f%%",
        startangle=90, colors=["#ff9999", "#66b3ff"],
    )
    ax.axis("equal")
    ax.set_title("Distribution des prédictions")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)  # libère la mémoire à chaque requête
    img_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render(
        request,
        "predictions/dashboard.html",
        {
            "total_submissions": total_submissions,
            "pneumonia_percentage": pneumonia_percentage,
            "submissions": latest,
            "img_data": img_data,
            "age_groups": dict(sorted(age_groups.items())),
        },
    )
