import time
from unittest import TestCase

from django.test import Client


class ViewTestCase(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_route(self):
        start = 'UCD, Dublin, Ireland'
        end = 'DCU, Dublin, Ireland'
        now = int(time.time())

        url = f'/routes/{start}/{end}/{now}'

        print(url)
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        for trip in data:
            for step in trip['steps']:
                if step['travel_mode'] == 'TRANSIT':
                    transit_details = step['transit_details']

                    self.assertIn('id', transit_details['arrival_stop'])
                    self.assertIn('id', transit_details['departure_stop'])
