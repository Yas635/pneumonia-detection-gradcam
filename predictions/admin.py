from django.contrib import admin

from .models import Submission


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    list_display = ("nom", "age", "prediction", "created_at")
    list_filter = ("prediction",)
