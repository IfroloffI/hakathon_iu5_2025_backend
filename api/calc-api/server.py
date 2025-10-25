import grpc
from concurrent import futures
import calc_pb2
import calc_pb2_grpc
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from poliastro.bodies import Sun, Earth
from poliastro.iod import izzo
from poliastro.util import norm


class CometCalculatorServicer(calc_pb2_grpc.CometCalculatorServicer):
    def CalculateOrbit(self, request, context):
        try:
            # Преобразуем наблюдения
            coords = []
            times = []
            for obs in request.observations:
                # RA в часах → градусы
                ra_deg = obs.ra_hours * 15
                coord = SkyCoord(ra=ra_deg * u.deg, dec=obs.dec_degrees * u.deg)
                coords.append(coord)
                times.append(Time(obs.timestamp, format="unix"))

            # Простейший пример: используем первые и последние наблюдения
            # В реальности нужно решать задачу определения орбиты (orbit determination)
            # Например, через метод Лапласа или Гаусса, или использовать poliastro + JPL

            # Для демонстрации возвращаем фиктивные данные

            response = calc_pb2.CalculateResponse(
                success=True,
                semi_major_axis_au=3.2,
                eccentricity=0.65,
                inclination_deg=12.5,
                longitude_ascending_node_deg=45.0,
                argument_perihelion_deg=80.0,
                perihelion_passage_jd=2460500.5,
                closest_approach_jd=2460600.5,
                closest_distance_au=0.8,
            )
            return response

        except Exception as e:
            return calc_pb2.CalculateResponse(success=False, error=str(e))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    calc_pb2_grpc.add_CometCalculatorServicer_to_server(
        CometCalculatorServicer(), server
    )
    server.add_insecure_port("[::]:50052")
    print("Python gRPC server running on port 50052")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
