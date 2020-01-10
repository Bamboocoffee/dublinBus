from datetime import date
from json import dumps

import googlemaps
import requests
from django.db import connection
from django.http import HttpResponse
from django.shortcuts import render
import json
from math import cos, sqrt
import sys
import os

# Create your views here.


def routes(request):
    gmap = googlemaps.Client(key='AIzaSyAf6-Rcx8RSnbzJELLh1dmpoBAOHh70Ax4')
    start = request.GET.get('start', '')
    stop = request.GET.get('stop', '')
    arr_time = request.GET.get('arr_time', '')
    dep_time = request.GET.get('dep_time', '')
    route_pref = request.GET.get('route_pref', '')

    data = gmap.directions(start, stop, mode='transit', transit_mode='bus',
                           alternatives=True, departure_time=dep_time, arrival_time=arr_time,
                           transit_routing_preference=route_pref)
    result = []

    for route in data:
        route_info = {
            "start_address": route['legs'][0]['start_address'],
            "end_address": route['legs'][0]['end_address'],
            "start_location": route['legs'][0]['start_location'],
            "end_location": route['legs'][0]['end_location'],
            "steps": route['legs'][0]['steps']
        }

        result.append(route_info)

    print(os.getcwd())
    json_data = open('/home/student/dublin_bus_app/web/static/stops_info.json')
    stops_data = json.load(json_data)

    busstops = json.loads(open("/home/student/dublin_bus_app/web/static/stops_info.json").read())

    def closestLocation(stop, busstops):
        def distance(lon1, lat1, lon2, lat2):
            R = 6371000  # radius of the Earth in m
            x = (lon2 - lon1) * cos(0.5 * (lat2 + lat1))
            y = (lat2 - lat1)
            return R * sqrt(x * x + y * y)

        buslist = sorted(busstops, key=lambda d: distance(d["longitude"], d["latitude"], stop['lng'], stop['lat']))
        return buslist[:1]

    for route in result:
        for step in route['steps']:
            if step['travel_mode'] == 'TRANSIT':
                route_id = step['transit_details']['line']['short_name'] \
                    if 'short_name' in step['transit_details']['line'] \
                    else step['transit_details']['line']['name']

                departure_stop = step['transit_details']['departure_stop']['location']
                arrival_stop = step['transit_details']['arrival_stop']['location']

                realDep = closestLocation(departure_stop, busstops)
                realArr = closestLocation(arrival_stop, busstops)

                step['transit_details']['arrival_stop']['id'] = realArr[0]["actual_stop_id"]
                step['transit_details']['departure_stop']['id'] = realDep[0]["actual_stop_id"]

    return HttpResponse(dumps(result), content_type='application/json')


# Returns all available route regrardless of schedule table
# def routes(request, start, end):
#     start_coord = search_place(start)
#     end_coord = search_place(end)
#
#     start_stops = search_nearby(start_coord)
#     end_stops = search_nearby(end_coord)
#     route_list = {}
#
#     for start_stop_id, start_route_name in start_stops.items():
#         for end_stop_id, end_route_name in end_stops.items():
#             route_set = start_route_name & end_route_name
#             if len(route_set) != 0:
#                 for route in route_set:
#                     route_list[route] = (start_stop_id, end_stop_id)
#
#     return HttpResponse(dumps(route_list), content_type='application/json')


# Get avaliable service for a given date
# There are might more than one serices returned
def services(year, month, day):
    try:
        day_of_week = date(year, month, day).strftime("%A").lower()
    except ValueError:
        return None

    with connection.cursor() as cursor:
        cursor.execute(f"select distinct sv.service_id from services sv where sv.{day_of_week} = 1")

        return [row for service in cursor.fetchall() for row in service]


# List all stop for a stop


# Returns all arrival times for a given bus route
def arrival_time(request, route, start_stop_id, end_stop_id):
    service = services(date.today().year, date.today().month, date.today().day)

    with connection.cursor() as cursor:
        cursor.execute("select distinct t.trip_id, cast(st.arrival_time as time) atime from routes r, trips t, "
                       "stop_times st, stops s where r.route_id = t.route_id and t.trip_id = st.trip_id and "
                       f"st.stop_id = s.stop_id and r.route_short_name = '{route}' and s.stop_id = '{start_stop_id}' "
                       f" and t.service_id in {tuple(service)} order by atime")

        return HttpResponse(dumps([(row[0], row[1].strftime("%H:%M:%S")) for row in cursor.fetchall()]),
                            content_type='application/json')


# def arrival_time(request, route, start_stop_id, end_stop_id):
#     pass


# Each route has a route id, but for the same route name such as 46a
# It many has more than one route id
# This is because for each route, it has two direction
# Also for each route, it has different schdule table, some route works on weekends, but others are not
# So to get a unique lists of stops for a route, the function need the route name, and the date
# def stops(request, trip_id):
# 	with connection.cursor() as cursor:
# 		cursor.execute("select s.stop_name, s.stop_lat, s.stop_lon, st.stop_sequence from stop_times st, "
# 		               f"stops s where st.stop_id = s.stop_id and st.trip_id = '{trip_id}' order by st.stop_sequence")
#
# 		data = [(row[0], str(row[1]), str(row[2])) for row in cursor.fetchall()]
#
# 	return HttpResponse(dumps(data), content_type='application/json')


# def stops(request, route_id, source, destination, time):
#     source_lat = float(source.split(',')[0])
#     source_lon = float(source.split(',')[1])
#     des_lat = float(destination.split(',')[0])
#     des_lon = float(destination.split(',')[1])
#
#     url = f'https://data.smartdublin.ie/cgi-bin/rtpi/routeinformation?routeid={route_id}&operator=bac'
#
#     r = requests.get(url)
#     result = {'routeid': None, 'routes': []}
#
#     destination = destination.split(',')[0].strip().lower()
#
#     if r.status_code == 200 and r.json()['errorcode'] == '0':
#         result['routeid'] = route_id,
#         for route in r.json()['results']:
#             result['routes'].append({
#                 'operator': route['operator'],
#                 'origin': route['origin'],
#                 'destination': route['destination'],
#                 'stops': [
#                     {
#                         'stopid': stop['stopid'],
#                         'shortname': stop['shortname'],
#                         'fullname': stop['fullname'],
#                         'latitude': stop['latitude'],
#                         'longitude': stop['longitude']
#                     } for stop in route['stops']]
#             })
#
#         final = {'routeid': result['routeid'], 'routes': []}
#
#         for route in result['routes']:
#             has_source = False
#             source_index = 0
#
#             has_des = False
#             des_index = 0
#
#             for index, stop in enumerate(route['stops']):
#                 lat = float(stop['latitude'])
#                 lon = float(stop['longitude'])
#
#                 if isclose(source_lat, lat, rel_tol=0.0001) and isclose(source_lon, lon, rel_tol=0.0001):
#                     source_index = index
#                     has_source = True
#
#                 if isclose(des_lat, lat, rel_tol=0.0001) and isclose(des_lon, lon, rel_tol=0.0001):
#                     des_index = index
#                     has_des = True
#
#             if has_source and has_des and source_index < des_index:
#                 final['routes'].append(route['stops'][source_index:des_index + 1])
#
#         final['routes'] = final['routes'][0]
#
#         return HttpResponse(dumps(final), content_type='application/json')

def stops(request, route, start, stop, time):
    start_lat, start_lon = start.split(',')
    stop_lat, stop_lon = stop.split(',')

    base_url = 'https://transit.api.here.com/v3/route.json'
    app_id = 'xh5WPe8LprSGbaeqW478'
    app_code = '5gav4wBVLuJdvs9CjYIL5g'

    url = f'{base_url}?app_id={app_id}&app_code={app_code}&dep={start_lat},{start_lon}&arr={stop_lat},{stop_lon}&time={time}&strict=1&modes=bus&max=1'

    r = requests.get(url)

    if r.status_code == 200:
        data = r.json()['Res']['Connections']['Connection'][0]['Sections']['Sec']
        result = None
        for trip in data:
            if trip['mode'] == 5:
                result = trip['Journey']['Stop']
        return HttpResponse(dumps(result), content_type='application/json')


# Data from google places api
def queryautocomplete(request, name):
    key = "AIzaSyAf6-Rcx8RSnbzJELLh1dmpoBAOHh70Ax4"
    url = f"https://maps.googleapis.com/maps/api/place/queryautocomplete/json?key={key}&input={name} Dublin,Ireland"

    res = requests.get(url)

    if res.status_code == 200:
        data = [place['description'] for place in res.json()['predictions']]

        return HttpResponse(dumps(data), content_type='application/json')


def search_place(place_name):
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?key=AIzaSyAf6" \
          f"-Rcx8RSnbzJELLh1dmpoBAOHh70Ax4&input={place_name}" \
          ",Dublin, Ireland&inputtype=textquery&fields=geometry&locationbias=ipbias"

    r = requests.get(url)

    if r.status_code == 200:
        data = r.json()

        if len(data['candidates']) != 0:
            geometry = data['candidates'][0]['geometry']
            return geometry['location']['lat'], geometry['location']['lng']


def search_nearby(coord):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?" \
          f"location={coord[0]},{coord[1]}&radius=5000&type=bus_station&" \
          f"key=AIzaSyAf6-Rcx8RSnbzJELLh1dmpoBAOHh70Ax4"

    r = requests.get(url)

    if r.status_code == 200:
        stop_names = [data['name'].replace("'", "''") for data in r.json()['results']]
        stop_name_filter = [f"s.stop_name ilike '{stop_name}'" for stop_name in stop_names]
        route_with_stop = {}

        with connection.cursor() as cursor:
            cursor.execute("select distinct s.stop_id from stops s where " + " or ".join(stop_name_filter))

            if cursor.rowcount != 0:
                stop_ids = [row[0] for row in cursor.fetchall()]
                stop_id_filter = [f"s.stop_id = '{stop_id}'" for stop_id in stop_ids]

                cursor.execute("select distinct r.route_short_name, s.stop_id from routes r, trips t, stop_times st, "
                               "stops s "
                               "where r.route_id = t.route_id and t.trip_id = st.trip_id and st.stop_id = s.stop_id "
                               f"and ({' or '.join(stop_id_filter)})")

                for row in cursor.fetchall():
                    value = route_with_stop.get(row[1], set())
                    value.add(row[0])
                    route_with_stop[row[1]] = value

                return route_with_stop


def index(request):
    return render(request, 'index.html')


def my_view(request):
    if request.user_agent.is_mobile:
        return render(request, 'm_index.html')
    elif request.user_agent.is_pc:
        return render(request, 'index.html')
    else:
        return render(request, 'test_index.html')


def payment_good(request):
    return render(request, 'payment_successful.html')


def payment_bad(request):
    return render(request, 'payment_unsuccessful.html')
