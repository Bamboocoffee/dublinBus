"""src URL Configuration

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
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path

from web.views import arrival_time, queryautocomplete
from web.views import routes
from web.views import stops
from web.views import index
from ml.views import predict
from web.views import payment_bad
from web.views import payment_good

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('payment_successful/', payment_good),
    path('payment_unsuccessful', payment_bad),
    # get the route name, route id and route direction from the
    # given source, destination and day of that month
    #
    # Arguments:
    # start: String - the starting point
    # end: String - the end point
    #
    # Returns:
    # {
    #   routes: [
    #               [route_name, start stop id, end stop id]
    #           ]
    # }
    path('routes', routes),
    # Lists the arrival times for each stops of a given route
    #
    # Arugments:
    # route: String - this is route id, get from the previous step
    # start_stop_id: String - the stop id of the source
    # end_stop_id: String - the stop id of the destination
    #
    # Returns:
    # [[trip_id, arrival_time], ...]
    # note: trip_id is very important because we will use it to show stops on the map later
    path('arrivals/<str:route>/<str:start_stop_id>/<str:end_stop_id>', arrival_time),
    # Returns stops for a given trip, a trip is determined by the route id,
    # route direction and the leave time
    # Arguments:
    # trip_id: String - the trip id return from last step
    # Returns:
    # [
    #   [stop_name, arrival_time]
    # ]
    path('stops/<str:route>/<str:start>/<str:stop>/<str:time>', stops),
    # Return a list of possible place when the users are typing in the input box
    #
    # Arguments:
    # trip_id: String - the trip id for the selected trip
    #
    # Returns:
    # [[stop name, lat, lon]]
    path('queryautocomplete/<str:name>', queryautocomplete),
    # Return a prediciton for a route at a particualr bus stop
    #
    # Arguments:
    # route_id: String
    #
    # Returns:
    #
    path('predict/<str:route_id>/<int:stop_id>/<int:planned_arrival>/<int:day_of_week>/<int:month>/<int:departure_time_seconds>/<str:t_minus_1118>', predict)
]

urlpatterns += staticfiles_urlpatterns()
