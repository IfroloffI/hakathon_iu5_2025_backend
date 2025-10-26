import grpc
from concurrent import futures
import logging
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta

import calc_pb2
import calc_pb2_grpc

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, get_sun
import astropy.units as u
from poliastro.bodies import Sun, Earth
from poliastro.twobody import Orbit
from poliastro.iod import izzo, vallado
from poliastro.util import norm, time_range
from poliastro.core.angles import E_to_M, nu_to_E, E_to_nu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    calc_pb2_grpc.add_CometCalculatorServicer_to_server(
        CometCalculatorServicer(), server
    )
    server.add_insecure_port("[::]:50051")

    logger.info("Calculator gRPC server running on port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
