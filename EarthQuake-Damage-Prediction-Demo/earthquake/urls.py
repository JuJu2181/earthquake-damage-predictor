from django.urls import path, include
from . import views

urlpatterns = [
    path('',views.homepage, name='homepage'),
    path('professional-user', views.professionalUser, name='professionalUser'),
    path('addprofessionalUserData', views.professionalUser_data, name='addprofessionalUserData')
]