import grpc
from concurrent import futures
import calc_pb2
import calc_pb2_grpc
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from poliastro.bodies import Sun, Earth
from poliastro.twobody import Orbit
from scipy.optimize import minimize


class CometCalculatorServicer(calc_pb2_grpc.CometCalculatorServicer):
    def CalculateOrbit(self, request, context):
        try:
            print("=" * 60)
            print("🔍 ВХОДЯЩИЕ ДАННЫЕ:")
            print(f"Количество наблюдений: {len(request.observations)}")
            
            # Логируем каждое наблюдение
            for i, obs in enumerate(request.observations):
                print(f"  Наблюдение {i+1}:")
                print(f"    RA: {obs.ra_hours:.6f} часов")
                print(f"    Dec: {obs.dec_degrees:.6f} градусов")
                print(f"    Timestamp: {obs.timestamp} (Unix)")
                
                # Логируем дополнительные поля если они есть
                if hasattr(obs, 'alt_degrees') and obs.alt_degrees:
                    print(f"    Alt: {obs.alt_degrees:.6f} градусов")
                if hasattr(obs, 'az_degrees') and obs.az_degrees:
                    print(f"    Az: {obs.az_degrees:.6f} градусов")
                if hasattr(obs, 'observer_lat_deg') and obs.observer_lat_deg:
                    print(f"    Observer Lat: {obs.observer_lat_deg:.6f} градусов")
                if hasattr(obs, 'observer_lon_deg') and obs.observer_lon_deg:
                    print(f"    Observer Lon: {obs.observer_lon_deg:.6f} градусов")
                if hasattr(obs, 'observer_height_m') and obs.observer_height_m:
                    print(f"    Observer Height: {obs.observer_height_m:.1f} м")
                if hasattr(obs, 'uncertainty_arcsec') and obs.uncertainty_arcsec:
                    print(f"    Uncertainty: {obs.uncertainty_arcsec:.3f} arcsec")
                print()
            
            # Логируем параметры запроса
            if hasattr(request, 'days_ahead') and request.days_ahead:
                print(f"Days ahead: {request.days_ahead}")
            else:
                print("Days ahead: не указан (используется значение по умолчанию 730)")
            
            if len(request.observations) < 5:
                print("❌ ОШИБКА: Нужно ≥5 наблюдений")
                return calc_pb2.CalculateResponse(
                    success=False,
                    error="At least 5 observations required"
                )
            
            print("🔄 Преобразуем наблюдения в формат для poliastro...")
            # Преобразуем наблюдения в формат для poliastro
            observations = []
            for obs in request.observations:
                coord = SkyCoord(
                    ra=obs.ra_hours * u.hourangle,
                    dec=obs.dec_degrees * u.deg,
                    distance=1.0 * u.AU  # начальное предположение
                )
                observations.append({
                    'time': Time(obs.timestamp, format='unix'),
                    'coord': coord,
                    'ra_hours': obs.ra_hours,
                    'dec_degrees': obs.dec_degrees
                })
            
            # Сортируем по времени
            observations.sort(key=lambda x: x['time'].jd)
            print(f"🔄 Отсортировано {len(observations)} наблюдений по времени")
            
            print("🔄 Начинаем определение орбиты методом наименьших квадратов...")
            # Определяем орбиту методом наименьших квадратов
            orbit_params = self._determine_orbit_poliastro(observations)
            
            print("🔄 Рассчитываем сближение с Землей...")
            # Рассчитываем сближение с Землей
            close_approach = self._calculate_close_approach(orbit_params['orbit'])
            
            # Формируем ответ
            response = calc_pb2.CalculateResponse(
                success=True,
                semi_major_axis_au=orbit_params['a'],
                eccentricity=orbit_params['e'],
                inclination_deg=orbit_params['i'],
                longitude_ascending_node_deg=orbit_params['raan'],
                argument_perihelion_deg=orbit_params['argp'],
                perihelion_passage_jd=orbit_params['t_p'],
                closest_approach_jd=close_approach['jd'],
                closest_distance_au=close_approach['distance_au'],
            )
            
            # Логируем результаты
            print("✅ РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ:")
            print(f"  Успех: {response.success}")
            print(f"  Большая полуось: {response.semi_major_axis_au:.6f} а.е.")
            print(f"  Эксцентриситет: {response.eccentricity:.6f}")
            print(f"  Наклонение: {response.inclination_deg:.6f} градусов")
            print(f"  Долгота восходящего узла: {response.longitude_ascending_node_deg:.6f} градусов")
            print(f"  Аргумент перицентра: {response.argument_perihelion_deg:.6f} градусов")
            print(f"  Прохождение перигелия: {response.perihelion_passage_jd:.6f} JD")
            print(f"  Ближайшее сближение: {response.closest_approach_jd:.6f} JD")
            print(f"  Расстояние при сближении: {response.closest_distance_au:.6f} а.е.")
            print("=" * 60)
            
            return response

        except Exception as e:
            print(f"❌ ОШИБКА ВЫЧИСЛЕНИЙ: {str(e)}")
            print("=" * 60)
            return calc_pb2.CalculateResponse(success=False, error=str(e))
    
    def _determine_orbit_poliastro(self, observations):
        """Определение орбиты с использованием poliastro"""
        
        def error_function(params):
            a, e, i, raan, argp, nu = params
            
            try:
                # Создаем тестовую орбиту
                test_orbit = Orbit.from_classical(
                    Sun,
                    a * u.AU,
                    e * u.one,
                    i * u.deg,
                    raan * u.deg,
                    argp * u.deg,
                    nu * u.deg,
                    observations[0]['time']
                )
                
                total_error = 0.0
                valid_observations = 0
                
                for obs in observations:
                    try:
                        # Предсказываем положение для времени наблюдения
                        time_from_epoch = obs['time'] - test_orbit.epoch
                        predicted_orbit = test_orbit.propagate(time_from_epoch)
                        
                        # Получаем положение Земли
                        earth_orbit = Orbit.from_body_ephem(Earth, obs['time'])
                        
                        # Вычисляем геоцентрическое положение
                        geo_position = predicted_orbit.r - earth_orbit.r
                        
                        # Преобразуем в небесные координаты
                        predicted_coord = SkyCoord(
                            x=geo_position[0], y=geo_position[1], z=geo_position[2],
                            representation_type='cartesian'
                        ).represent_as('spherical')
                        
                        # Вычисляем угловую ошибку
                        angular_separation = obs['coord'].separation(predicted_coord)
                        total_error += angular_separation.deg ** 2
                        valid_observations += 1
                        
                    except Exception as e:
                        continue
                
                if valid_observations == 0:
                    return 1e10
                    
                return total_error / valid_observations
                
            except Exception:
                return 1e10
        
        # Начальное приближение (параметры Марса)
        x0 = np.array([1.52, 0.09, 1.85, 49.56, 286.5, 0.0])
        print(f"🔄 Начальные параметры орбиты: a={x0[0]:.3f}, e={x0[1]:.3f}, i={x0[2]:.3f}°")
        
        # Оптимизация
        print("🔄 Запускаем оптимизацию методом Nelder-Mead...")
        result = minimize(error_function, x0, method='Nelder-Mead', 
                         options={'maxiter': 200})
        
        print(f"🔄 Оптимизация завершена. Итераций: {result.nit}, Функций: {result.nfev}")
        print(f"🔄 Финальная ошибка: {result.fun:.6f}")
        
        a, e, i, raan, argp, nu = result.x
        print(f"🔄 Финальные параметры: a={a:.6f}, e={e:.6f}, i={i:.6f}°, Ω={raan:.6f}°, ω={argp:.6f}°")
        
        # Создаем финальную орбиту
        orbit = Orbit.from_classical(
            Sun,
            a * u.AU,
            e * u.one,
            i * u.deg,
            raan * u.deg,
            argp * u.deg,
            nu * u.deg,
            observations[0]['time']
        )
        
        # Время перигелия (упрощенно)
        t_p = orbit.epoch.jd
        
        return {
            'a': a, 'e': e, 'i': i, 'raan': raan, 'argp': argp, 't_p': t_p,
            'orbit': orbit
        }
    
    def _calculate_close_approach(self, orbit, days_range=730):
        """Расчет ближайшего сближения с Землей"""
        print(f"🔄 Ищем ближайшее сближение в течение {days_range} дней...")
        min_distance = float('inf')
        closest_time = orbit.epoch
        checked_points = 0
        
        # Создаем свой time_range
        for days in np.linspace(0, days_range, 100):
            time = orbit.epoch + days * u.day
            try:
                # Положение кометы
                comet_orbit = orbit.propagate(time - orbit.epoch)
                
                # Положение Земли
                earth_orbit = Orbit.from_body_ephem(Earth, time)
                
                # Расстояние
                distance = np.linalg.norm(
                    (comet_orbit.r - earth_orbit.r).to(u.AU).value
                )
                
                checked_points += 1
                
                if distance < min_distance:
                    min_distance = distance
                    closest_time = time
                    
            except Exception:
                continue
        
        print(f"🔄 Проверено {checked_points} точек. Минимальное расстояние: {min_distance:.6f} а.е.")
        print(f"🔄 Время сближения: {closest_time.iso}")
        
        return {
            'jd': closest_time.jd,
            'distance_au': min_distance
        }


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    calc_pb2_grpc.add_CometCalculatorServicer_to_server(
        CometCalculatorServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    print("🚀 Advanced Comet Calculator with poliastro running on port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()