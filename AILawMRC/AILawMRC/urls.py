"""AILawMRC URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from user import views as uv

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('user/register', uv.register),
    path('user/login', uv.login),
    path('user/updatePwd', uv.updatePwd),
    path('user/updateInfo', uv.updateInfo),

    path('user/crawl', uv.crawl),
    path('user/crawlRate', uv.crawlRate),

    path('user/uploadfile', uv.uploadfile),

    path('user/getcardcr', uv.getcardcr),
    path('user/removecardcr', uv.removecardcr),
    path('user/getcrawlrecord', uv.getcrawlrecord),
    path('user/getcardq', uv.getcardq),
    path('user/removecardq', uv.removecardq),
    path('user/getfiles', uv.getfiles),
    path('user/getanalyzerecord', uv.getanalyzerecord),

    path('user/analyzeCR', uv.analyzeCR),

    path('user/readcomprehend', uv.readcomprehend),
    path('user/change', uv.change),
    path('user/getanswer', uv.getanswer)
]
