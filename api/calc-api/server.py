"""
Comet orbit determination gRPC server (RA/Dec only, per given proto).
- Input: Observation { ra_hours, dec_degrees, timestamp }
- Output: 6 orbital elements + closest approach (JD, AU)
- No horizontal coordinates used.
- Earth positions computed via Astropy (get_body_barycentric).
- Graceful shutdown on SIGINT/SIGTERM.
"""

import grpc
from concurrent import futures
import threading
import signal
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation, get_body_barycentric
from astropy.time import Time
from scipy.optimize import least_squares

from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from poliastro.util import norm

import calc_pb2
import calc_pb2_grpc


# ===========================
# Data model for observations
# ===========================
@dataclass
class Obs:
    ra_hours: float
    dec_deg: float
    ts_unix: float


# ===========================
# Helpers
# ===========================
def unitvec_to_radec(uvec: np.ndarray) -> Tuple[float, float]:
    """Direction vector (ICRS) -> (RA, Dec) in degrees."""
    c = SkyCoord(x=uvec[0], y=uvec[1], z=uvec[2],
                 representation_type="cartesian", frame="icrs")
    return (c.ra.deg, c.dec.deg)


def earth_heliocentric_positions(times: List[Time]) -> List[np.ndarray]:
    """
    Sun->Earth heliocentric vectors (ICRS) in AU using Astropy only.
    Compatible with poliastro 0.7.x (no Ephem class needed).
    """
    rs = []
    for t in times:
        r_earth_bary = get_body_barycentric('earth', t)
        r_sun_bary = get_body_barycentric('sun', t)
        r_helio = (r_earth_bary.xyz - r_sun_bary.xyz).to(u.AU).value  # Sun -> Earth
        rs.append(np.array(r_helio).flatten())
    return rs


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Minimal signed difference between two angles in degrees, range [-180, 180)."""
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


# ===========================
# Orbit fitting (least squares)
# ===========================
def residuals_elements(p, times: List[Time], re_list: List[np.ndarray],
                       obs_radec_deg: List[Tuple[float, float]], t0: Time):
    """
    p = [a(AU), e, i(deg), raan(deg), argp(deg), t_peri(JD)]
    Returns residual vector [dRA1, dDec1, dRA2, dDec2, ...] in degrees.
    """
    a_AU, e, i_deg, raan_deg, argp_deg, t_peri_jd = p

    epoch = t0.tdb
    k = Sun.k
    a = a_AU * u.AU

    # keep optimizer in a sane region
    if not (0.1 < a_AU < 100.0) or not (0.0 <= e < 0.999999) or not (0.0 <= i_deg <= 180.0):
        return 1e3 * np.ones(len(times) * 2)

    i = i_deg * u.deg
    raan = raan_deg * u.deg
    argp = argp_deg * u.deg
    t_peri = Time(t_peri_jd, format="jd", scale="tdb")

    # Mean anomaly at epoch
    n = np.sqrt(k / a**3)  # rad/s
    M0 = (n * (epoch - t_peri).to(u.s)).to(u.rad).value

    def mean_to_true(M, ecc):
        E = M
        for _ in range(60):
            f = E - ecc * np.sin(E) - M
            fp = 1 - ecc * np.cos(E)
            E -= f / fp
        cos_nu = (np.cos(E) - ecc) / (1 - ecc * np.cos(E))
        sin_nu = (np.sqrt(1 - ecc**2) * np.sin(E)) / (1 - ecc * np.cos(E))
        return np.arctan2(sin_nu, cos_nu)

    nu0 = mean_to_true(M0, e) * u.rad

    try:
        orb0 = Orbit.from_classical(Sun, a, e * u.one, i, raan, argp, nu0, epoch=epoch)
    except Exception:
        return 1e3 * np.ones(len(times) * 2)

    res = []
    for idx, (t, re_icrs) in enumerate(zip(times, re_list)):
        dt = (t.tdb - epoch).to(u.s)
        orb_t = orb0.propagate(dt)
        rc = orb_t.r.to(u.AU).value  # Sun -> Comet
        udir = (rc - re_icrs)
        udir /= np.linalg.norm(udir)  # Earth -> Comet unit vector

        ra_calc, dec_calc = unitvec_to_radec(udir)
        ra_obs, dec_obs = obs_radec_deg[idx]

        # RA wrapped difference, scaled by cos(Dec) to balance
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

    # Initial guess and bounds (physical ranges)
    p0 = np.array([3.0, 0.5, 10.0, 30.0, 60.0, t0.tdb.jd])
    lo = np.array([0.1, 0.0, 0.0,   0.0,   0.0,   t0.tdb.jd - 5000.0])
    hi = np.array([100, 0.999999, 180.0, 360.0, 360.0, t0.tdb.jd + 5000.0])

    fun = lambda p: residuals_elements(p, times, re_list, obs_radec_deg, t0)
    lsq = least_squares(fun, p0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=600)

    a_AU, e, i_deg, raan_deg, argp_deg, t_peri_jd = lsq.x

    # Build orbit object at epoch for later propagation
    k = Sun.k
    a = a_AU * u.AU
    n = np.sqrt(k / a**3)
    M0 = (n * (t0.tdb - Time(t_peri_jd, format="jd", scale="tdb")).to(u.s)).to(u.rad).value

    def mean_to_true(M, ecc):
        E = M
        for _ in range(60):
            f = E - ecc * np.sin(E) - M
            fp = 1 - ecc * np.cos(E)
            E -= f / fp
        cos_nu = (np.cos(E) - ecc) / (1 - ecc * np.cos(E))
        sin_nu = (np.sqrt(1 - ecc**2) * np.sin(E)) / (1 - ecc * np.cos(E))
        return np.arctan2(sin_nu, cos_nu)

    nu0 = mean_to_true(M0, e) * u.rad
    orbit = Orbit.from_classical(
        Sun,
        a, e * u.one,
        i_deg * u.deg, raan_deg * u.deg, argp_deg * u.deg,
        nu0,
        epoch=t0.tdb
    )

    elements = dict(
        a_AU=float(a_AU),
        e=float(e),
        i_deg=float(i_deg),
        raan_deg=float(raan_deg % 360.0),
        argp_deg=float(argp_deg % 360.0),
        t_peri_jd=float(t_peri_jd),
    )
    return orbit, elements, t0


# ===========================
# Closest approach search
# ===========================
def closest_approach(orbit: Orbit, start_utc: Time, days: int = 1488) -> Tuple[Time, float]:
    """
    Coarse grid search of minimal Earth-Comet distance on [start, start+days].
    Uses Astropy barycentric positions for Earth/Sun each step.
    """
    ts = start_utc.tdb + np.linspace(0, days, 2000) * u.day  # coarse grid
    epoch = orbit.epoch

    best_idx = 0
    best_d = float("inf")
    for idx, tt in enumerate(ts):
        dt = (tt - epoch).to(u.s)
        orb_t = orbit.propagate(dt)
        rc = orb_t.r.to(u.AU).value  # Sun->Comet

        r_earth_bary = get_body_barycentric('earth', tt)
        r_sun_bary = get_body_barycentric('sun', tt)
        r_earth = (r_earth_bary.xyz - r_sun_bary.xyz).to(u.AU).value  # Sun->Earth

        d = float(norm(rc - r_earth))
        if d < best_d:
            best_d = d
            best_idx = idx

    return ts[best_idx], best_d


# ===========================
# gRPC service
# ===========================
class CometCalculatorServicer(calc_pb2_grpc.CometCalculatorServicer):
    def CalculateOrbit(self, request, context):
        try:
            if len(request.observations) < 5:
                return calc_pb2.CalculateResponse(success=False, error="Нужно ≥5 наблюдений")

            obs_list = [
                Obs(o.ra_hours, o.dec_degrees, o.timestamp)
                for o in sorted(request.observations, key=lambda r: r.timestamp)
            ]

            orbit, elems, _t0 = estimate_orbit(obs_list)
            t_min, d_min_au = closest_approach(
                orbit,
                start_utc=Time(obs_list[0].ts_unix, format="unix", scale="utc")
                # 'days' not in proto -> use function default (1488)
            )

            return calc_pb2.CalculateResponse(
                success=True,
                error="",
                semi_major_axis_au=elems["a_AU"],
                eccentricity=elems["e"],
                inclination_deg=elems["i_deg"],
                longitude_ascending_node_deg=elems["raan_deg"],
                argument_perihelion_deg=elems["argp_deg"],
                perihelion_passage_jd=elems["t_peri_jd"],
                closest_approach_jd=t_min.jd,
                closest_distance_au=d_min_au,
            )

        except Exception as e:
            return calc_pb2.CalculateResponse(success=False, error=str(e))


# ===========================
# Server bootstrap with graceful shutdown
# ===========================
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    calc_pb2_grpc.add_CometCalculatorServicer_to_server(CometCalculatorServicer(), server)
    server.add_insecure_port("[::]:50051")

    stop_event = threading.Event()

    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        stop_event.set()
        server.stop(grace=30)  # wait up to 30s for inflight RPCs
        print("Server stopped cleanly.")

    signal.signal(signal.SIGINT, handle_signal)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # docker/systemd

    print("🚀 gRPC server running on :50051 (RA/Dec only). Press Ctrl+C to stop.")
    server.start()

    try:
        stop_event.wait()
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)


if __name__ == "__main__":
    serve()
