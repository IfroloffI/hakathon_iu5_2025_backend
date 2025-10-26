"""
Comet orbit determination gRPC server.
Совместим с Python 3.13 и poliastro<=0.7.0
(без использования poliastro.ephem.Ephem)

poliastro.ephem.Ephem v0.17.0-v0.20.0 совместим с python3.8-10
"""

import grpc
from concurrent import futures
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from astropy import units as u
from astropy.coordinates import (
    SkyCoord, CartesianRepresentation,
    EarthLocation, AltAz, get_body_barycentric
)
from astropy.time import Time

from scipy.optimize import least_squares
from poliastro.bodies import Sun, Earth
from poliastro.twobody import Orbit
from poliastro.util import norm

import calc_pb2
import calc_pb2_grpc

import threading
import signal


# ==========================================================
# Данные наблюдений
# ==========================================================
@dataclass
class Obs:
    ra_hours: float
    dec_deg: float
    ts_unix: float


# ==========================================================
# Утилиты
# ==========================================================
def radec_unitvec(ra_deg: float, dec_deg: float) -> np.ndarray:
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    rep: CartesianRepresentation = c.represent_as(CartesianRepresentation)
    return np.array([rep.x.value, rep.y.value, rep.z.value])


def unitvec_to_radec(uvec: np.ndarray) -> Tuple[float, float]:
    c = SkyCoord(x=uvec[0], y=uvec[1], z=uvec[2],
                 representation_type="cartesian", frame="icrs")
    return (c.ra.deg, c.dec.deg)


def earth_heliocentric_positions(times: List[Time]) -> List[np.ndarray]:
    """
    Возвращает список гелиоцентрических векторов Sun->Earth в а.е.
    без использования poliastro.ephem.Ephem (совместимо с poliastro 0.7+)
    """
    rs = []
    for t in times:
        r_earth_bary = get_body_barycentric('earth', t)
        r_sun_bary = get_body_barycentric('sun', t)
        # Гелиоцентрический вектор: Sun -> Earth
        r_helio = (r_earth_bary.xyz - r_sun_bary.xyz).to(u.AU).value
        rs.append(np.array(r_helio).flatten())
    return rs


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Минимальная разница углов в градусах [-180, 180)"""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return d


def altaz_to_radec(
    alt_deg: float, az_deg: float,
    lat_deg: float, lon_deg: float, height_m: float,
    ts_unix: float
) -> Tuple[float, float]:
    """Преобразование горизонтальных координат в экваториальные (ICRS)"""
    t = Time(ts_unix, format="unix", scale="utc")
    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=height_m * u.m)
    altaz = AltAz(alt=alt_deg * u.deg, az=az_deg * u.deg, obstime=t, location=loc)
    icrs = SkyCoord(altaz).transform_to("icrs")
    return icrs.ra.hour, icrs.dec.deg


# ==========================================================
# Основные расчёты орбиты
# ==========================================================
def residuals_elements(p, times: List[Time], re_list: List[np.ndarray],
                       obs_radec_deg: List[Tuple[float, float]], t0: Time):
    a_AU, e, i_deg, raan_deg, argp_deg, t_peri_jd = p
    epoch = t0.tdb

    k = Sun.k
    a = a_AU * u.AU
    i = i_deg * u.deg
    raan = raan_deg * u.deg
    argp = argp_deg * u.deg
    t_peri = Time(t_peri_jd, format="jd", scale="tdb")

    n = np.sqrt(k / a**3)
    M0 = (n * (epoch - t_peri).to(u.s)).to(u.rad).value

    def mean_to_true(M, e):
        E = M
        for _ in range(50):
            E -= (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        cos_nu = (np.cos(E) - e) / (1 - e*np.cos(E))
        sin_nu = (np.sqrt(1 - e**2)*np.sin(E)) / (1 - e*np.cos(E))
        return np.arctan2(sin_nu, cos_nu)

    nu0 = mean_to_true(M0, e) * u.rad

    try:
        orb0 = Orbit.from_classical(Sun, a, e*u.one, i, raan, argp, nu0, epoch=epoch)
    except Exception:
        return 1e3 * np.ones(len(times)*2)

    res = []
    for idx, (t, re_icrs) in enumerate(zip(times, re_list)):
        dt = (t.tdb - epoch).to(u.s)
        orb_t = orb0.propagate(dt)
        rc = orb_t.r.to(u.AU).value
        udir = (rc - re_icrs) / np.linalg.norm(rc - re_icrs)
        ra_calc, dec_calc = unitvec_to_radec(udir)
        ra_obs, dec_obs = obs_radec_deg[idx]
        d_ra = angle_diff_deg(ra_calc, ra_obs) * np.cos(np.deg2rad(dec_obs))
        d_dec = dec_calc - dec_obs
        res.extend([d_ra, d_dec])
    return np.array(res)


def estimate_orbit(observations: List[Obs]):
    if len(observations) < 5:
        raise ValueError("Нужно ≥ 5 наблюдений.")

    obs_sorted = sorted(observations, key=lambda o: o.ts_unix)
    times = [Time(o.ts_unix, format="unix", scale="utc") for o in obs_sorted]
    t0 = times[0]

    obs_radec_deg = [(o.ra_hours * 15.0, o.dec_deg) for o in obs_sorted]
    re_list = earth_heliocentric_positions(times)

    p0 = np.array([3.0, 0.5, 10.0, 30.0, 60.0, t0.tdb.jd])
    lo = np.array([0.1, 0.0, 0.0, 0.0, 0.0, t0.tdb.jd - 5000])
    hi = np.array([100, 0.999, 180.0, 360.0, 360.0, t0.tdb.jd + 5000])

    fun = lambda p: residuals_elements(p, times, re_list, obs_radec_deg, t0)
    lsq = least_squares(fun, p0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)

    a_AU, e, i_deg, raan_deg, argp_deg, t_peri_jd = lsq.x
    rms = float(np.sqrt(np.mean(lsq.fun**2)))

    orbit = Orbit.from_classical(
        Sun, a_AU*u.AU, e*u.one, i_deg*u.deg, raan_deg*u.deg,
        argp_deg*u.deg, 0*u.deg, epoch=t0.tdb
    )
    elems = dict(
        a_AU=float(a_AU),
        e=float(e),
        i_deg=float(i_deg),
        raan_deg=float(raan_deg % 360),
        argp_deg=float(argp_deg % 360),
        t_peri_jd=float(t_peri_jd),
        rms_deg=rms
    )
    return orbit, elems


def closest_approach(orbit: Orbit, start_utc: Time, days: int = 1488):
    """
    Грубый поиск ближайшего сближения кометы с Землёй на интервале days.
    Использует astropy.get_body_barycentric вместо Ephem.
    """
    ts = start_utc.tdb + np.linspace(0, days, 2000)*u.day
    comet_rs = orbit.sample(ts)
    dists = []
    for t, rc in zip(ts, comet_rs):
        r_earth_bary = get_body_barycentric('earth', t)
        r_sun_bary = get_body_barycentric('sun', t)
        r_earth = (r_earth_bary.xyz - r_sun_bary.xyz).to(u.AU).value
        d = norm((rc.to(u.AU).value - r_earth))
        dists.append(d)
    k0 = int(np.argmin(dists))
    return ts[k0], float(dists[k0])


# ==========================================================
# gRPC сервис
# ==========================================================
class CometCalculatorServicer(calc_pb2_grpc.CometCalculatorServicer):
    def CalculateOrbit(self, request, context):
        try:
            if len(request.observations) < 5:
                return calc_pb2.CalculateResponse(success=False, error="Нужно ≥5 наблюдений")

            obs_list = []
            for o in request.observations:
                ra, dec = o.ra_hours, o.dec_degrees
                if (o.alt_degrees or o.az_degrees) and (o.observer_lat_deg or o.observer_lon_deg):
                    ra, dec = altaz_to_radec(
                        o.alt_degrees, o.az_degrees,
                        o.observer_lat_deg, o.observer_lon_deg, o.observer_height_m,
                        o.timestamp
                    )
                obs_list.append(Obs(ra, dec, o.timestamp))

            orbit, elems = estimate_orbit(obs_list)
            t_min, d_min_au = closest_approach(
                orbit,
                start_utc=Time(obs_list[0].ts_unix, format="unix", scale="utc"),
                days=request.days_ahead or 400
            )

            return calc_pb2.CalculateResponse(
                success=True,
                semi_major_axis_au=elems["a_AU"],
                eccentricity=elems["e"],
                inclination_deg=elems["i_deg"],
                longitude_ascending_node_deg=elems["raan_deg"],
                argument_perihelion_deg=elems["argp_deg"],
                perihelion_passage_jd=elems["t_peri_jd"],
                rms_residual_deg=elems["rms_deg"],
                closest_approach_jd=t_min.jd,
                closest_distance_au=d_min_au,
            )

        except Exception as e:
            return calc_pb2.CalculateResponse(success=False, error=str(e))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    calc_pb2_grpc.add_CometCalculatorServicer_to_server(CometCalculatorServicer(), server)
    server.add_insecure_port("[::]:50051")

    stop_event = threading.Event()

    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        stop_event.set()
        # Завершает сервер с ожиданием всех активных запросов
        server.stop(grace=30)  # подождать до 30 секунд
        print("Server stopped cleanly.")

    # Подписываемся на сигналы ОС
    signal.signal(signal.SIGINT, handle_signal)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # systemd/docker stop

    print("🚀 Python gRPC server running on port 50051 (Press Ctrl+C to stop)")
    server.start()

    # Ждём завершения сигнала
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)

if __name__ == "__main__":
    serve()
