from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("", include("Back_Front.urls")),
    path('admin/', admin.site.urls),
]