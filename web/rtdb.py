import psycopg2
import requests

base_url = 'https://data.smartdublin.ie/cgi-bin/rtpi/'


def get_all_routes():
    url = f'{base_url}routelistinformation?operator=bac'

    r = requests.get(url)

    if r.status_code == 200:
        routes = [route['route'] for route in r.json()['results'] if '|' not in route['route']]

        return list(set(routes))


def get_stops(route):
    url = f'{base_url}routeinformation?routeid={route}&operator=bac'

    r = requests.get(url)

    stops = set()

    if r.status_code == 200:
        for trip in r.json()['results']:
            for stop in trip['stops']:
                stops.add((route, stop['stopid'], float(stop['latitude']), float(stop['longitude'])))

        return list(stops)


def save_to_database():
    routes = get_all_routes()

    connection = psycopg2.connect(user='postgres', password='postgres', host='127.0.0.1',
                                  port='5432', database='postgres')
    cursor = connection.cursor()

    for route in routes:
        stops = get_stops(route)
        print('route: ' + route)
        for stop in stops:
            cursor.execute('INSERT INTO rt_stops VALUES (%s, %s, %s, %s)', stop)

        print(f'route: {route} inserted')

    connection.commit()

    cursor.close()
    connection.close()


if __name__ == '__main__':
    save_to_database()
