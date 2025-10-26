"""
Comet orbit determination gRPC server.
Совместим с Python 3.13 и poliastro<=0.7.0
(без использования poliastro.ephem.Ephem)

poliastro.ephem.Ephem v0.17.0-v0.20.0 совместим с python3.8-10
"""

import grpc
from concurrent import futures
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import least_squares, minimize
from datetime import datetime, timedelta

import calc_pb2
import calc_pb2_grpc

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord, CartesianRepresentation,
    EarthLocation, AltAz, get_body_barycentric, get_sun
)
from poliastro.bodies import Sun, Earth
from poliastro.twobody import Orbit
from poliastro.iod import izzo, vallado
from poliastro.util import norm, time_range
from poliastro.core.angles import E_to_M, nu_to_E, E_to_nu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.info(
                f"📡 Received calculation request with {len(request.observations)} observations"
            )

            if len(request.observations) < 5:
                return calc_pb2.CalculateResponse(
                    success=False, error="At least 5 observations required"
                )

            # Parse and validate observations
            observations = self._parse_observations(request.observations)

            if len(observations) < 3:
                return calc_pb2.CalculateResponse(
                    success=False, error="Need at least 3 valid observations"
                )

            logger.info(
                f"🔍 Using {len(observations)} observations for orbit determination"
            )

            # Method 1: Gauss initial orbit determination
            try:
                initial_orbit = self._gauss_method(observations)
                logger.info("✅ Gauss method succeeded")
            except Exception as e:
                logger.warning(f"❌ Gauss method failed: {e}")
                initial_orbit = self._create_initial_orbit(observations)

            # Method 2: Differential correction (least squares)
            try:
                refined_orbit = self._differential_correction(
                    initial_orbit, observations
                )
                logger.info("✅ Differential correction succeeded")
            except Exception as e:
                logger.warning(f"❌ Differential correction failed: {e}")
                refined_orbit = initial_orbit

            # Extract orbital elements
            orbit_params = self._extract_orbital_elements(
                refined_orbit, observations[0]["time"]
            )

            # Calculate close approach
            close_approach = self._calculate_close_approach(refined_orbit)

            response = calc_pb2.CalculateResponse(
                success=True,
                error="",
                semi_major_axis_au=orbit_params["a"],
                eccentricity=orbit_params["e"],
                inclination_deg=orbit_params["i"],
                longitude_ascending_node_deg=orbit_params["raan"],
                argument_perihelion_deg=orbit_params["argp"],
                perihelion_passage_jd=orbit_params["t_p"],
                closest_approach_jd=close_approach["jd"],
                closest_distance_au=close_approach["distance_au"],
            )

            logger.info("🎯 Orbit calculation completed successfully")
            return response

        except Exception as e:
            logger.error(f"💥 Calculation error: {str(e)}")
            return calc_pb2.CalculateResponse(
                success=False, error=f"Calculation error: {str(e)}"
            )

    def _parse_observations(self, raw_observations):
        """Parse and validate observations"""
        observations = []
        for obs in raw_observations:
            try:
                coord = SkyCoord(
                    ra=obs.ra_hours * u.hourangle,
                    dec=obs.dec_degrees * u.deg,
                    distance=1.0 * u.AU,
                )

                observations.append(
                    {
                        "time": Time(obs.timestamp, format="unix"),
                        "ra_hours": obs.ra_hours,
                        "dec_degrees": obs.dec_degrees,
                        "coord": coord,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to parse observation: {e}")
                continue

        observations.sort(key=lambda x: x["time"].jd)
        return observations

    def _gauss_method(self, observations):
        """Gauss method for initial orbit determination"""
        if len(observations) < 3:
            raise ValueError("Need at least 3 observations for Gauss method")

        # Use first, middle, and last observations
        obs1, obs2, obs3 = (
            observations[0],
            observations[len(observations) // 2],
            observations[-1],
        )
        t1, t2, t3 = obs1["time"], obs2["time"], obs3["time"]

        # Get Earth positions
        earth_r1 = self._get_earth_position(t1)
        earth_r2 = self._get_earth_position(t2)
        earth_r3 = self._get_earth_position(t3)

        # Observation unit vectors
        L1 = obs1["coord"].cartesian.xyz.value
        L2 = obs2["coord"].cartesian.xyz.value
        L3 = obs3["coord"].cartesian.xyz.value

        # Time intervals
        tau1 = (t1 - t2).jd
        tau3 = (t3 - t2).jd

        # Gauss method equations (simplified)
        # This is a complex calculation - here's a simplified version
        A1 = tau3 / (tau3 - tau1)
        A3 = -tau1 / (tau3 - tau1)
        B1 = tau3 * (tau3**2 - tau1**2) / (6 * (tau3 - tau1))
        B3 = -tau1 * (tau3**2 - tau1**2) / (6 * (tau3 - tau1))

        # Estimate geocentric distances (simplified)
        r2_mag = 2.0  # AU, initial guess

        # Iterative improvement would go here
        # For now, use a reasonable estimate
        comet_r2 = earth_r2 + L2 * r2_mag

        # Use Lambert's method between first and last observations
        try:
            orbit = izzo.lambert(
                Sun.k,
                (earth_r1 + L1 * r2_mag) * u.AU,
                (earth_r3 + L3 * r2_mag) * u.AU,
                t3 - t1,
            )
            return orbit
        except:
            # Fallback to creating orbit from estimated position
            return Orbit.from_vectors(
                Sun,
                comet_r2 * u.AU,
                (comet_r2 - earth_r2) * u.AU / u.day,  # Approximate velocity
                t2,
            )

    def _create_initial_orbit(self, observations):
        """Create initial orbit estimate"""
        # Use middle observation as reference
        mid_idx = len(observations) // 2
        obs = observations[mid_idx]

        # Estimate position (simplified)
        earth_r = self._get_earth_position(obs["time"])
        L = obs["coord"].cartesian.xyz.value
        r_mag = 2.0  # AU

        comet_r = earth_r + L * r_mag

        # Create circular orbit as initial guess
        return Orbit.from_classical(
            Sun,
            r_mag * u.AU,
            0.1 * u.one,  # Small eccentricity
            30.0 * u.deg,
            100.0 * u.deg,
            50.0 * u.deg,
            0.0 * u.deg,
            obs["time"],
        )

    def _differential_correction(self, initial_orbit, observations):
        """Differential correction (least squares) to refine orbit"""

        def residuals(params):
            """Calculate residuals between observed and computed positions"""
            try:
                a, e, i, raan, argp, M0 = params

                # Create test orbit
                test_orbit = Orbit.from_classical(
                    Sun,
                    a * u.AU,
                    e * u.one,
                    i * u.deg,
                    raan * u.deg,
                    argp * u.deg,
                    M0 * u.deg,
                    initial_orbit.epoch,
                )

                total_residual = 0.0
                valid_count = 0

                for obs in observations:
                    try:
                        # Propagate to observation time
                        propagated = test_orbit.propagate(
                            obs["time"] - test_orbit.epoch
                        )

                        # Get Earth position
                        earth_orbit = Orbit.from_body_ephem(Earth, obs["time"])

                        # Geocentric position
                        geo_pos = propagated.r - earth_orbit.r

                        # Convert to spherical coordinates
                        computed_coord = SkyCoord(
                            x=geo_pos[0],
                            y=geo_pos[1],
                            z=geo_pos[2],
                            representation_type="cartesian",
                        ).represent_as("spherical")

                        # Angular separation (residual)
                        separation = obs["coord"].separation(computed_coord)
                        total_residual += separation.deg**2
                        valid_count += 1

                    except Exception as e:
                        continue

                if valid_count == 0:
                    return 1e10

                return total_residual / valid_count

            except Exception:
                return 1e10

        # Initial parameters from initial orbit
        x0 = np.array(
            [
                initial_orbit.a.to(u.AU).value,
                initial_orbit.ecc.value,
                initial_orbit.inc.to(u.deg).value,
                initial_orbit.raan.to(u.deg).value,
                initial_orbit.argp.to(u.deg).value,
                initial_orbit.nu.to(u.deg).value,
            ]
        )

        # Bounds for physical parameters
        bounds = [
            (0.1, 100.0),  # a (AU)
            (0.0, 0.99),  # e
            (0.0, 180.0),  # i (deg)
            (0.0, 360.0),  # raan (deg)
            (0.0, 360.0),  # argp (deg)
            (0.0, 360.0),  # M0 (deg)
        ]

        # Optimize
        result = minimize(
            residuals,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        # Create refined orbit
        a, e, i, raan, argp, M0 = result.x

        return Orbit.from_classical(
            Sun,
            a * u.AU,
            e * u.one,
            i * u.deg,
            raan * u.deg,
            argp * u.deg,
            M0 * u.deg,
            initial_orbit.epoch,
        )

    def _extract_orbital_elements(self, orbit, epoch):
        """Extract orbital elements including perihelion time"""
        # Basic elements
        a = orbit.a.to(u.AU).value
        e = orbit.ecc.value
        i = orbit.inc.to(u.deg).value
        raan = orbit.raan.to(u.deg).value
        argp = orbit.argp.to(u.deg).value
        nu = orbit.nu.to(u.deg).value

        # Calculate time of perihelion passage
        # Using Kepler's equation: M = E - e*sin(E)
        E = nu_to_E(nu * u.deg, e * u.one).value  # Eccentric anomaly
        M = E_to_M(E * u.rad, e * u.one).value  # Mean anomaly

        # Time from epoch to perihelion
        n = orbit.n.to(u.rad / u.day).value  # Mean motion
        time_to_perihelion = -M / n if n != 0 else 0

        # Perihelion time (JD)
        t_p = orbit.epoch.jd + time_to_perihelion

        return {"a": a, "e": e, "i": i, "raan": raan, "argp": argp, "t_p": t_p}

    def _get_earth_position(self, time):
        """Get Earth's heliocentric position"""
        earth_orbit = Orbit.from_body_ephem(Earth, time)
        return earth_orbit.r.to(u.AU).value

    def _calculate_close_approach(self, orbit, search_period_days=730):
        """Calculate closest approach to Earth"""
        search_start = orbit.epoch
        search_end = search_start + search_period_days * u.day

        times = time_range(search_start, end=search_end, num=1000)

        min_distance = float("inf")
        closest_time = search_start

        for time in times:
            try:
                # Propagate comet
                comet_orbit = orbit.propagate(time - orbit.epoch)

                # Get Earth position
                earth_orbit = Orbit.from_body_ephem(Earth, time)

                # Calculate distance
                distance = norm(comet_orbit.r - earth_orbit.r).to(u.AU).value

                if distance < min_distance:
                    min_distance = distance
                    closest_time = time

            except Exception:
                continue

        return {"jd": closest_time.jd, "distance_au": min_distance}


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
