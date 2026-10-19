import sys
import time
import pytest
from simple_pid import PID


def test_zero():
    pid = PID(1, 1, 1, setpoint=0)
    assert pid(0) == 0


def test_P():
    pid = PID(1, 0, 0, setpoint=10, sample_time=None)
    assert pid(0) == 10
    assert pid(5) == 5
    assert pid(-5) == 15


def test_P_negative_setpoint():
    pid = PID(1, 0, 0, setpoint=-10, sample_time=None)
    assert pid(0) == -10
    assert pid(5) == -15
    assert pid(-5) == -5
    assert pid(-15) == 5


def test_I():
    pid = PID(0, 10, 0, setpoint=10, sample_time=0.1)
    time.sleep(0.1)

    assert round(pid(0)) == 10.0  # Make sure we are close to expected value
    time.sleep(0.1)

    assert round(pid(0)) == 20.0


def test_I_negative_setpoint():
    pid = PID(0, 10, 0, setpoint=-10, sample_time=0.1)
    time.sleep(0.1)

    assert round(pid(0)) == -10.0
    time.sleep(0.1)

    assert round(pid(0)) == -20.0


def test_D():
    pid = PID(0, 0, 0.1, setpoint=10, sample_time=0.1)

    # Should not compute derivative when there is no previous input (don't assume 0 as first input)
    assert pid(0) == 0
    time.sleep(0.1)

    # Derivative is 0 when input is the same
    assert pid(0) == 0
    assert pid(0) == 0
    time.sleep(0.1)

    assert round(pid(5)) == -5
    time.sleep(0.1)
    assert round(pid(15)) == -10


def test_D_negative_setpoint():
    pid = PID(0, 0, 0.1, setpoint=-10, sample_time=0.1)
    time.sleep(0.1)

    # Should not compute derivative when there is no previous input (don't assume 0 as first input)
    assert pid(0) == 0
    time.sleep(0.1)

    # Derivative is 0 when input is the same
    assert pid(0) == 0
    assert pid(0) == 0
    time.sleep(0.1)

    assert round(pid(5)) == -5
    time.sleep(0.1)
    assert round(pid(-5)) == 10
    time.sleep(0.1)
    assert round(pid(-15)) == 10


@pytest.mark.parametrize('on_measurement', [True, False])
@pytest.mark.parametrize('gain', [2, -2])
def test_derivative_filter_response(on_measurement, gain):
    pid = PID(
        0,
        0,
        gain,
        sample_time=None,
        derivative_filter_tau=0.5,
        differential_on_measurement=on_measurement,
    )
    assert pid(10, dt=0.1) == 0  # No derivative kick on the first input

    # A step produces a smaller peak than the raw derivative (-gain / 0.5).
    assert pid(11, dt=0.5) == pytest.approx(-gain)
    assert pid.components[2] == pytest.approx(-gain)
    # A constant input lets the filter decay, using each actual timestep.
    assert pid(11, dt=0.5) == pytest.approx(-gain / 2)
    assert pid(11, dt=1.5) == pytest.approx(-gain / 8)


@pytest.mark.parametrize('on_measurement, expected', [(True, 0), (False, 4)])
def test_derivative_filter_setpoint_change(on_measurement, expected):
    pid = PID(
        0,
        0,
        2,
        sample_time=None,
        derivative_filter_tau=0.5,
        differential_on_measurement=on_measurement,
    )
    pid(0, dt=0.5)
    pid.setpoint = 2
    assert pid(0, dt=0.5) == pytest.approx(expected)


def test_derivative_filter_noise_attenuation():
    raw = PID(0, 0, 1, sample_time=None)
    filtered = PID(0, 0, 1, sample_time=None, derivative_filter_tau=0.1)
    raw_outputs, filtered_outputs = [], []
    for input_ in [1, -1] * 50:
        raw_outputs.append(raw(input_, dt=0.01))
        filtered_outputs.append(filtered(input_, dt=0.01))
    assert max(map(abs, filtered_outputs)) < max(map(abs, raw_outputs)) / 10


def test_derivative_filter_disabled():
    default = PID(1, 2, 3, sample_time=None)
    disabled = PID(1, 2, 3, sample_time=None, derivative_filter_tau=0)
    filtered = PID(1, 2, 3, sample_time=None, derivative_filter_tau=0.5)
    for input_, dt in [(10, 0.1), (11, 0.5), (9, 0.2), (9, 1)]:
        assert default(input_, dt=dt) == disabled(input_, dt=dt)
        filtered(input_, dt=dt)
        assert default.components == disabled.components
        assert filtered.components[:2] == default.components[:2]
    # Disabling filtering while running immediately uses the raw derivative.
    filtered.derivative_filter_tau = 0
    assert filtered(8, dt=0.5) == default(8, dt=0.5)


def test_derivative_filter_sample_time_and_clock():
    now = 0.0
    pid = PID(0, 0, 2, sample_time=0.5, time_fn=lambda: now, derivative_filter_tau=0.5)
    assert pid(10) == 0
    now = 0.5
    assert pid(11) == pytest.approx(-2)
    now = 0.75
    assert pid(100) == pytest.approx(-2)  # Skipped input must not affect the filter
    now = 1.0
    assert pid(11) == pytest.approx(-1)


@pytest.mark.parametrize('auto_mode_reset', [True, False])
def test_derivative_filter_reset(auto_mode_reset):
    pid = PID(0, 0, 2, sample_time=None, derivative_filter_tau=0.5)
    pid(10, dt=0.5)
    assert pid(11, dt=0.5) == pytest.approx(-2)
    if auto_mode_reset:
        pid.auto_mode = False
        assert pid(100, dt=0.5) == pytest.approx(-2)
        assert pid.components[2] == pytest.approx(-2)
        pid.auto_mode = True
    else:
        pid.reset()
    assert pid.components[2] == 0
    assert pid.derivative_filter_tau == 0.5
    assert pid(100, dt=0.5) == 0
    assert pid(101, dt=0.5) == pytest.approx(-2)


def test_derivative_filter_output_limits():
    pid = PID(0, 0, 2, sample_time=None, derivative_filter_tau=0.5, output_limits=(-1, 1))
    assert pid(10, dt=0.5) == 0
    assert pid(11, dt=0.5) == -1
    assert pid.components[2] == pytest.approx(-2)
    assert pid(11, dt=0.5) == pytest.approx(-1)


@pytest.mark.parametrize('tau', [-1, float('nan'), float('inf'), -float('inf')])
def test_derivative_filter_invalid_tau(tau):
    with pytest.raises(ValueError, match='derivative_filter_tau'):
        PID(derivative_filter_tau=tau)
    pid = PID(derivative_filter_tau=0.5)
    with pytest.raises(ValueError, match='derivative_filter_tau'):
        pid.derivative_filter_tau = tau
    assert pid.derivative_filter_tau == 0.5


def test_desired_state():
    pid = PID(10, 5, 2, setpoint=10, sample_time=None)

    # Should not make any adjustment when setpoint is achieved
    assert pid(10) == 0


def test_output_limits():
    pid = PID(100, 20, 40, setpoint=10, output_limits=(0, 100), sample_time=None)
    time.sleep(0.1)

    assert 0 <= pid(0) <= 100
    time.sleep(0.1)

    assert 0 <= pid(-100) <= 100


def test_sample_time():
    pid = PID(setpoint=10, sample_time=10)

    control = pid(0)

    # Last value should be returned again
    assert pid(100) == control


def test_time_fn():
    pid = PID()

    # Default time function should be time.monotonic, or time.time in older versions of Python
    if sys.version_info < (3, 3):
        assert pid.time_fn == time.time
    else:
        assert pid.time_fn == time.monotonic

    i = 0

    def time_function():
        nonlocal i
        i += 1
        return i

    pid.time_fn = time_function

    for j in range(1, 5):
        # Call pid a few times and verify that the time function above was used
        pid(0)
        assert pid._last_time == j


def test_time_fn_notime():
    # Deliberately prevent the time module from being imported
    import sys

    sys.modules['time'] = None

    with pytest.raises(ModuleNotFoundError):
        # Must specify a time_fn if time is not available
        _ = PID()

    # We can still create a PID if we specify our own time_fn
    _ = PID(time_fn=lambda: 0)

    # Restore time module so the following tests can use it
    sys.modules['time'] = time


def test_starting_output():
    # If the PID is started with a system already at the setpoint, we can give it our best guess
    # for which output it should start at
    pid = PID(1, 0, 0, setpoint=10, starting_output=25)
    assert pid(10) == 25


def test_auto_mode():
    pid = PID(1, 0, 0, setpoint=10, sample_time=None)

    # Ensure updates happen by default
    assert pid(0) == 10
    assert pid(5) == 5

    # Ensure no new updates happen when auto mode is off
    pid.auto_mode = False
    assert pid(1) == 5
    assert pid(7) == 5

    # Should reset when reactivating
    pid.auto_mode = True
    assert pid._last_input is None
    assert pid._integral == 0
    assert pid(8) == 2

    # Last update time should be reset to avoid huge dt
    pid.auto_mode = False
    time.sleep(1)
    pid.auto_mode = True
    assert pid.time_fn() - pid._last_time < 0.01

    # Check that setting last_output works
    pid.auto_mode = False
    pid.set_auto_mode(True, last_output=10)
    assert pid._integral == 10


def test_separate_components():
    pid = PID(1, 0, 1, setpoint=10, sample_time=0.1)

    assert pid(0) == 10
    assert pid.components == (10, 0, 0)
    time.sleep(0.1)

    assert round(pid(5)) == -45
    assert tuple(round(term) for term in pid.components) == (5, 0, -50)


def test_clamp():
    from simple_pid.pid import _clamp

    assert _clamp(None, (None, None)) is None
    assert _clamp(None, (-10, 10)) is None

    # No limits
    assert _clamp(0, (None, None)) == 0
    assert _clamp(100, (None, None)) == 100
    assert _clamp(-100, (None, None)) == -100

    # Only lower limit
    assert _clamp(0, (0, None)) == 0
    assert _clamp(100, (0, None)) == 100
    assert _clamp(-100, (0, None)) == 0

    # Only upper limit
    assert _clamp(0, (None, 0)) == 0
    assert _clamp(100, (None, 0)) == 0
    assert _clamp(-100, (None, 0)) == -100

    # Both limits
    assert _clamp(0, (-10, 10)) == 0
    assert _clamp(-10, (-10, 10)) == -10
    assert _clamp(10, (-10, 10)) == 10
    assert _clamp(-100, (-10, 10)) == -10
    assert _clamp(100, (-10, 10)) == 10


def test_repr():
    pid = PID(1, 2, 3, setpoint=10, derivative_filter_tau=0.5)
    new_pid = eval(repr(pid))

    assert new_pid.Kp == 1
    assert new_pid.Ki == 2
    assert new_pid.Kd == 3
    assert new_pid.setpoint == 10
    assert new_pid.derivative_filter_tau == 0.5


def test_converge_system():
    pid = PID(1, 0.8, 0.04, setpoint=5, output_limits=(-5, 5))
    pv = 0  # Process variable

    def update_system(c, dt):
        # Calculate a simple system model
        return pv + c * dt - 1 * dt

    start_time = time.time()
    last_time = start_time

    while time.time() - start_time < 120:
        c = pid(pv)
        pv = update_system(c, time.time() - last_time)

        last_time = time.time()

    # Check if system has converged
    assert abs(pv - 5) < 0.1


def test_converge_diff_on_error():
    pid = PID(1, 0.8, 0.04, setpoint=5, output_limits=(-5, 5), differential_on_measurement=False)
    pv = 0  # Process variable

    def update_system(c, dt):
        # Calculate a simple system model
        return pv + c * dt - 1 * dt

    start_time = time.time()
    last_time = start_time

    while time.time() - start_time < 12:
        c = pid(pv)
        pv = update_system(c, time.time() - last_time)

        last_time = time.time()

    # Check if system has converged
    assert abs(pv - 5) < 0.1


def test_error_map():
    import math

    def pi_clip(angle):
        """Transform the angle value to a [-pi, pi) range."""
        if angle > 0:
            if angle > math.pi:
                return angle - 2 * math.pi
        else:
            if angle < -math.pi:
                return angle + 2 * math.pi
        return angle

    sp = 0.0  # Setpoint
    pv = 5.0  # Process variable
    pid = PID(1, 0, 0, setpoint=0.0, sample_time=0.1, error_map=pi_clip)

    # Check if error value is mapped by the function
    assert pid(pv) == pi_clip(sp - pv)
