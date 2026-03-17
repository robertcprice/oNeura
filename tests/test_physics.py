"""
Physics Module Tests

Comprehensive tests for the oNeura physics module, all runnable
without MuJoCo installed. Tests cover:
- LIF neuron dynamics
- ConnectomeBridge signal propagation
- PhysicsEnvironment creation and MJCF generation
- BrainMotorInterface sensor encoding
- State vector completeness (tarsus joints)
"""

import numpy as np
import pytest

from oneuro.physics.connectome_bridge import (
    ConnectomeBridge,
    NeuronPool,
    NeuronType,
    SynapseModel,
    SynapseType,
)
from oneuro.physics.brain_motor_interface import (
    BrainMotorInterface,
    SensorEncoder,
    NeuralPatternDecoder,
    MotorPrimitiveType,
)
from oneuro.physics.drosophila_simulator import (
    DrosophilaPhysicsState,
    MotorPatterns,
    LegID,
)
from oneuro.physics.physics_environment import (
    PhysicsEnvironment,
    Fruit,
    Plant,
    Obstacle,
)
from oneuro.physics.compound_eye import (
    CompoundEye,
    BinocularVisionSystem,
    create_visual_encoder_output,
)


# ============================================================
# LIF Neuron Tests
# ============================================================

class TestNeuronPool:
    """Tests for leaky integrate-and-fire neuron pool."""

    def test_single_neuron_fires(self):
        """A neuron with sufficient input must produce spikes."""
        pool = NeuronPool("test", NeuronType.INTERNEURON, 10)
        total_spikes = 0
        for t in range(100):
            spikes = pool.update(np.full(10, 5.0), dt=1.0, time=float(t))
            total_spikes += spikes.sum()
        assert total_spikes > 0, "Neurons should fire with strong input"

    def test_neuron_stays_silent(self):
        """A neuron with zero input should not fire."""
        pool = NeuronPool("test", NeuronType.INTERNEURON, 10)
        total_spikes = 0
        for t in range(100):
            spikes = pool.update(np.zeros(10), dt=1.0, time=float(t))
            total_spikes += spikes.sum()
        assert total_spikes == 0, "Neurons should not fire with zero input"

    def test_refractory_period(self):
        """Neuron should not spike during refractory period."""
        pool = NeuronPool("test", NeuronType.MOTOR, 1, refractory_period=5.0)
        # Drive hard to cause a spike
        spike_times = []
        for t in range(50):
            spikes = pool.update(np.array([10.0]), dt=1.0, time=float(t))
            if spikes[0] > 0:
                spike_times.append(t)

        # Check inter-spike intervals respect refractory period
        for i in range(1, len(spike_times)):
            isi = spike_times[i] - spike_times[i - 1]
            assert isi >= 5, f"ISI {isi} violates refractory period of 5ms"

    def test_firing_rate_estimate(self):
        """Firing rate should track actual spike count."""
        pool = NeuronPool("test", NeuronType.INTERNEURON, 5)
        for t in range(200):
            pool.update(np.full(5, 5.0), dt=1.0, time=float(t))
        rates = pool.get_activity()
        assert rates.shape == (5,)
        assert np.any(rates > 0), "Firing rate should be positive after sustained input"

    def test_input_padding(self):
        """Input smaller than pool size should be zero-padded."""
        pool = NeuronPool("test", NeuronType.INTERNEURON, 10)
        # Only provide 3 values — should pad remaining 7 with zeros
        spikes = pool.update(np.array([5.0, 5.0, 5.0]), dt=1.0, time=0.0)
        assert spikes.shape == (10,)

    def test_membrane_reset_after_spike(self):
        """After spiking, membrane potential resets to resting."""
        pool = NeuronPool("test", NeuronType.INTERNEURON, 1)
        # Drive hard
        for t in range(10):
            spikes = pool.update(np.array([20.0]), dt=1.0, time=float(t))
            if spikes[0] > 0:
                assert pool.membrane_potentials[0] == pool.resting_potential
                break


# ============================================================
# ConnectomeBridge Tests
# ============================================================

class TestConnectomeBridge:
    """Tests for connectome bridge signal routing."""

    def test_pool_creation(self):
        """All 13 neuron pools should be created."""
        bridge = ConnectomeBridge(num_neurons_per_pool=10)
        assert len(bridge.pools) == 13
        assert "mechanosensory_tarsi" in bridge.pools
        assert "leg_motor_front" in bridge.pools

    def test_connections_created(self):
        """Default connectivity should create synapses."""
        bridge = ConnectomeBridge(num_neurons_per_pool=10)
        assert len(bridge.connections) > 0

    def test_all_connections_have_pool_names(self):
        """Every synapse must store pre_pool and post_pool names."""
        bridge = ConnectomeBridge(num_neurons_per_pool=10)
        for conn in bridge.connections:
            assert isinstance(conn.pre_pool, str)
            assert isinstance(conn.post_pool, str)
            assert conn.pre_pool in bridge.pools
            assert conn.post_pool in bridge.pools

    def test_activity_propagates(self):
        """Sensory input should reach motor pools within 500ms."""
        bridge = ConnectomeBridge(num_neurons_per_pool=50)
        bridge.set_sensory_input("mechanosensory_tarsi", np.ones(50) * 0.8)
        bridge.set_sensory_input("proprioceptive_legs", np.ones(50) * 0.6)

        for _ in range(50):
            activities = bridge.step(dt=1.0)

        motor_activity = sum(
            activities[p].sum()
            for p in ["leg_motor_front", "leg_motor_middle", "leg_motor_hind"]
        )
        assert motor_activity > 0, "Motor pools should show activity"

    def test_motor_pools_activate(self):
        """All 5 motor pools should show some activity."""
        bridge = ConnectomeBridge(num_neurons_per_pool=50)
        bridge.set_sensory_input("mechanosensory_tarsi", np.ones(50))
        bridge.set_sensory_input("proprioceptive_legs", np.ones(50))
        bridge.set_sensory_input("visual_motion", np.ones(50) * 0.5)

        for _ in range(100):
            bridge.step(dt=1.0)

        motor_output = bridge.get_motor_output()
        assert len(motor_output) == 5
        active_pools = sum(1 for v in motor_output.values() if v.sum() > 0)
        # At least leg motor pools should be active
        assert active_pools >= 3, f"Only {active_pools}/5 motor pools active"

    def test_cross_inhibition_works(self):
        """Inhibitory connections should suppress target pools."""
        bridge = ConnectomeBridge(num_neurons_per_pool=50)
        # Find an inhibitory connection
        inhibitory = [c for c in bridge.connections if c.synapse_type == SynapseType.INHIBITORY]
        assert len(inhibitory) > 0, "Should have inhibitory connections"
        # All inhibitory weights should be negative
        for conn in inhibitory:
            assert conn.weight < 0, "Inhibitory weights must be negative"

    def test_plasticity_updates_weights(self):
        """Hebbian learning should modify weights when enabled."""
        bridge = ConnectomeBridge(num_neurons_per_pool=20, enable_plasticity=True)

        # Set plasticity rate on some connections
        for conn in bridge.connections[:10]:
            conn.plasticity_rate = 0.01

        initial_weights = [c.weight for c in bridge.connections[:10]]

        bridge.set_sensory_input("mechanosensory_tarsi", np.ones(20))
        for _ in range(50):
            bridge.step(dt=1.0)

        final_weights = [c.weight for c in bridge.connections[:10]]
        changed = sum(1 for i, f in zip(initial_weights, final_weights) if abs(i - f) > 1e-6)
        assert changed > 0, "Plasticity should change at least some weights"

    def test_brain_output_vector(self):
        """get_brain_output_vector should return flattened motor activity."""
        bridge = ConnectomeBridge(num_neurons_per_pool=10)
        output = bridge.get_brain_output_vector()
        assert output.shape == (50,)  # 5 motor pools * 10 neurons

    def test_reset(self):
        """Reset should restore all pools to resting state."""
        bridge = ConnectomeBridge(num_neurons_per_pool=10)
        bridge.set_sensory_input("mechanosensory_tarsi", np.ones(10))
        bridge.step(dt=1.0)
        bridge.reset()

        assert bridge.time == 0.0
        for pool in bridge.pools.values():
            assert np.allclose(pool.membrane_potentials, pool.resting_potential)
            assert np.allclose(pool.firing_rates, 0.0)

    def test_sensory_input_validation(self):
        """Setting input on non-sensory pool should raise."""
        bridge = ConnectomeBridge(num_neurons_per_pool=10)
        with pytest.raises(ValueError):
            bridge.set_sensory_input("leg_motor_front", np.ones(10))


# ============================================================
# PhysicsEnvironment Tests
# ============================================================

class TestPhysicsEnvironment:
    """Tests for 3D physics environment."""

    def test_environment_creation(self):
        """Environment should store objects correctly."""
        env = PhysicsEnvironment(arena_size=(8.0, 8.0))
        env.add_fruit(pos=[1, 2, 0.1], sugar=0.8)
        env.add_plant(pos=[0, -1], height=2.5)
        env.add_obstacle(pos=[1, 1, 0.3], size=[0.5, 0.5, 0.6], shape="box")
        assert len(env.fruits) == 1
        assert len(env.plants) == 1
        assert len(env.obstacles) == 1
        assert env.fruits[0].sugar_concentration == 0.8

    def test_mjcf_generation(self):
        """Generated MJCF should be valid XML with environment objects."""
        env = PhysicsEnvironment(arena_size=(6.0, 6.0))
        env.add_fruit(pos=[1, 1, 0.1])
        env.add_plant(pos=[0, 0], height=2.0)
        env.add_obstacle(pos=[2, 2, 0.2], size=[0.3, 0.3, 0.4])

        xml_str = env.generate_mjcf()
        assert len(xml_str) > 0

        # Verify it parses as valid XML
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

        # Check objects are present
        assert "fruit_0" in xml_str
        assert "plant_0" in xml_str
        assert "obstacle_0" in xml_str
        assert "wall_north" in xml_str

    def test_foraging_arena_preset(self):
        """Predefined foraging arena should have expected objects."""
        env = PhysicsEnvironment.foraging_arena()
        assert len(env.fruits) == 3
        assert len(env.plants) == 2
        assert len(env.obstacles) == 2
        assert env.wind_speed == 0.5

    def test_obstacle_course_preset(self):
        """Obstacle course should have dense obstacles."""
        env = PhysicsEnvironment.obstacle_course()
        assert len(env.obstacles) >= 7

    def test_empty_arena_preset(self):
        """Empty arena should have no objects."""
        env = PhysicsEnvironment.empty_arena(size=5.0)
        assert len(env.fruits) == 0
        assert len(env.plants) == 0
        assert len(env.obstacles) == 0
        assert env.arena_size == (5.0, 5.0)

    def test_environment_sampling(self):
        """Distance-based odorant should return sensible values."""
        env = PhysicsEnvironment()
        env.add_fruit(pos=[0, 0, 0], sugar=1.0)

        # Close to fruit → high odorant
        near = env.sample_environment(np.array([0.0, 0.0, 0.0]))
        assert near["odorant"] > 0.5

        # Far from fruit → low odorant
        far = env.sample_environment(np.array([10.0, 10.0, 0.0]))
        assert far["odorant"] < near["odorant"]

    def test_temperature_gradient(self):
        """Temperature should decrease with altitude."""
        env = PhysicsEnvironment()
        env.temperature_base = 25.0
        env.temperature_gradient = -2.0

        low = env.sample_environment(np.array([0, 0, 0]))
        high = env.sample_environment(np.array([0, 0, 5]))
        assert high["temperature"] < low["temperature"]

    def test_wind_force_computation(self):
        """Wind should produce force in wind direction."""
        env = PhysicsEnvironment()
        env.set_wind(direction=[1, 0, 0], speed=1.0)

        # Stationary fly
        force = env.compute_wind_force(np.zeros(3))
        assert force[0] > 0, "Wind force should push in wind direction"
        assert abs(force[1]) < 1e-10
        assert abs(force[2]) < 1e-10

    def test_wind_zero_when_disabled(self):
        """No wind → no force."""
        env = PhysicsEnvironment()
        force = env.compute_wind_force(np.zeros(3))
        np.testing.assert_array_almost_equal(force, np.zeros(3))

    def test_nearest_fruit(self):
        """Should find correct nearest fruit."""
        env = PhysicsEnvironment()
        env.add_fruit(pos=[10, 10, 0])
        env.add_fruit(pos=[1, 0, 0])
        env.add_fruit(pos=[5, 5, 0])

        idx, dist = env.get_nearest_fruit(np.zeros(3))
        assert idx == 1  # [1, 0, 0] is closest
        assert dist == pytest.approx(1.0, abs=0.01)

    def test_nearest_fruit_empty(self):
        """No fruits → return None."""
        env = PhysicsEnvironment()
        assert env.get_nearest_fruit(np.zeros(3)) is None

    def test_repr(self):
        """String representation should include counts."""
        env = PhysicsEnvironment.foraging_arena()
        s = repr(env)
        assert "fruits=3" in s
        assert "plants=2" in s


# ============================================================
# BrainMotorInterface Tests
# ============================================================

class TestSensorEncoder:
    """Tests for sensor encoding including tarsus."""

    def test_encoder_includes_tarsus(self):
        """Tarsus joints must be in the encoded proprioceptive vector."""
        encoder = SensorEncoder()
        joint_positions = {
            "Ta1_L_pitch": 0.2,
            "Ta2_L_pitch": 0.15,
            "Ta3_L_pitch": 0.1,
            "Ta1_R_pitch": 0.2,
            "Ta2_R_pitch": 0.15,
            "Ta3_R_pitch": 0.1,
        }
        pattern = encoder.encode(
            joint_positions=joint_positions,
            joint_velocities={},
            touch_contacts={},
        )
        # The pattern should not be all zeros if we provided tarsus data
        proprioceptive_section = pattern[:encoder.num_proprioceptive]
        assert np.any(proprioceptive_section > 0), "Tarsus data should appear in encoding"

    def test_output_dimension(self):
        """Output dimension should be proprioceptive + touch + visual."""
        encoder = SensorEncoder()
        assert encoder.output_dim == 48 + 6 + 64  # 118

    def test_encode_full_state(self):
        """Encoding a full state should produce correct-dimension output."""
        encoder = SensorEncoder()
        joint_pos = {f"C1_L_pitch": 0.3, f"F1_L_pitch": 0.8}
        pattern = encoder.encode(
            joint_positions=joint_pos,
            joint_velocities={},
            touch_contacts={},
        )
        assert pattern.shape == (encoder.output_dim,)


class TestNeuralPatternDecoder:
    """Tests for neural pattern decoder."""

    def test_decode_returns_tuple(self):
        """Decode should return (primitive_type, intensity, modulations)."""
        decoder = NeuralPatternDecoder()
        pattern = np.random.randn(32)
        result = decoder.decode(pattern)
        assert len(result) == 3
        ptype, intensity, mods = result
        assert isinstance(ptype, MotorPrimitiveType)
        assert 0.0 <= intensity <= 1.0
        assert isinstance(mods, dict)

    def test_decode_handles_wrong_size(self):
        """Decoder should handle input of wrong dimension."""
        decoder = NeuralPatternDecoder(num_output_neurons=32)
        # Too small
        result = decoder.decode(np.zeros(5))
        assert len(result) == 3
        # Too large
        result = decoder.decode(np.zeros(100))
        assert len(result) == 3


class TestBrainMotorInterface:
    """Tests for the main brain-motor interface."""

    def test_reflex_controller_runs(self):
        """Reflex controller should not crash."""
        interface = BrainMotorInterface()
        state = DrosophilaPhysicsState()
        state.joint_positions = {"C1_L_pitch": 0.3}
        state.joint_velocities = {"C1_L_pitch": 0.0}
        state.touch_contacts = {LegID.LEFT_FRONT: True}
        commands = interface.process(state, dt=0.001)
        assert isinstance(commands, list)

    def test_full_pipeline_no_mujoco(self):
        """Encode → decode → commands pipeline should work without MuJoCo."""
        encoder = SensorEncoder()
        decoder = NeuralPatternDecoder()
        interface = BrainMotorInterface(
            sensor_encoder=encoder,
            pattern_decoder=decoder,
        )
        state = DrosophilaPhysicsState()
        state.joint_positions = {
            "C1_L_pitch": 0.3, "F1_L_pitch": 0.8, "T1_L_pitch": -0.5,
            "Ta1_L_pitch": 0.2,
        }
        state.touch_contacts = {
            LegID.LEFT_FRONT: True, LegID.LEFT_MIDDLE: True,
            LegID.LEFT_HIND: True,
        }
        commands = interface.process(state, dt=0.001)
        assert isinstance(commands, list)


# ============================================================
# State Vector Tests
# ============================================================

class TestStateVector:
    """Tests for DrosophilaPhysicsState vector encoding."""

    def test_state_vector_includes_tarsus(self):
        """State vector should include all 24 leg joint positions."""
        state = DrosophilaPhysicsState()
        # Set all 24 joints
        tarsus_joints = [
            "Ta1_L_pitch", "Ta2_L_pitch", "Ta3_L_pitch",
            "Ta1_R_pitch", "Ta2_R_pitch", "Ta3_R_pitch",
        ]
        for jname in tarsus_joints:
            state.joint_positions[jname] = 0.1

        vec = state.to_vector()
        # 3 (pos) + 4 (quat) + 3 (vel) + 3 (angvel) + 24 (joints) + 2 (wings) + 6 (touch) = 45
        assert len(vec) == 45

    def test_state_vector_roundtrip(self):
        """Calling to_vector twice should give identical result."""
        state = DrosophilaPhysicsState()
        state.body_position = np.array([1.0, 2.0, 3.0])
        state.joint_positions["C1_L_pitch"] = 0.5
        state.joint_positions["Ta1_L_pitch"] = 0.2

        vec1 = state.to_vector()
        vec2 = state.to_vector()
        np.testing.assert_array_equal(vec1, vec2)

    def test_state_vector_dimension_consistency(self):
        """observation_dim should match to_vector length."""
        state = DrosophilaPhysicsState()
        assert state.observation_dim == len(state.to_vector())


# ============================================================
# Motor Patterns Tests
# ============================================================

class TestMotorPatterns:
    """Tests for predefined motor patterns."""

    def test_stance_includes_tarsus(self):
        """Stance phase should include tarsus joint targets."""
        stance = MotorPatterns.stance_phase(LegID.LEFT_FRONT)
        assert "Ta1_L_pitch" in stance
        assert stance["Ta1_L_pitch"] == pytest.approx(0.2)

    def test_swing_includes_tarsus(self):
        """Swing phase should include tarsus joint targets."""
        swing = MotorPatterns.swing_phase(LegID.LEFT_FRONT)
        assert "Ta1_L_pitch" in swing
        assert swing["Ta1_L_pitch"] == pytest.approx(-0.3)

    def test_tripod_gait_has_all_legs(self):
        """Tripod gait should produce targets for all legs."""
        targets = MotorPatterns.tripod_gait(0.25)
        # Should have entries for all 6 legs (4 joints each = 24 entries)
        assert len(targets) == 24

    def test_wingbeat_pattern(self):
        """Wingbeat should produce opposing wing angles."""
        wb = MotorPatterns.wingbeat(250.0, 0.001)
        assert "wing_L" in wb
        assert "wing_R" in wb
        # Wings should be roughly opposite in sign
        assert wb["wing_L"] * wb["wing_R"] <= 0 or abs(wb["wing_L"]) < 0.01


class TestCompoundEye:
    """Test compound eye visual system."""

    def test_eye_creation(self):
        """Test compound eye initialization."""
        eye = CompoundEye(eye_side="left", num_ommatidia=768)
        assert eye.eye_side == "left"
        # Hexagonal lattice generates approximately the requested number
        assert len(eye.ommatidia) > 100  # Should have many ommatidia
        assert len(eye.ommatidia) <= 768  # But not more than requested

    def test_ommatidia_coverage(self):
        """Test that ommatidia cover visual field."""
        eye = CompoundEye(eye_side="left")

        # Check coverage
        azims = [om.azimuth for om in eye.ommatidia]
        elevs = [om.elevation for om in eye.ommatidia]

        # Should cover hemisphere with reasonable extent
        assert min(azims) < -np.pi/4  # At least 45 degrees to the side
        assert max(azims) > np.pi/4
        assert min(elevs) < -np.pi/12  # At least 15 degrees up/down
        assert max(elevs) > np.pi/12

    def test_light_sampling(self):
        """Test light intensity sampling."""
        eye = CompoundEye(eye_side="left")

        # Light from above
        light_dir = np.array([0, 0, 1])
        intensities = eye.sample_environment(
            light_direction=light_dir,
            light_intensity=1.0,
            body_orientation=np.array([1, 0, 0, 0]),
        )

        assert len(intensities) == 768
        # Upper ommatidia should have higher intensity
        assert np.mean([intensities[i] for i, om in enumerate(eye.ommatidia) if om.elevation > 0]) > \
               np.mean([intensities[i] for i, om in enumerate(eye.ommatidia) if om.elevation < 0])

    def test_motion_detectors(self):
        """Test Hassenstein-Reichardt EMDs."""
        eye = CompoundEye(eye_side="left")

        # Should have horizontal and vertical motion detectors
        assert len(eye.motion_detectors_h) > 0
        assert len(eye.motion_detectors_v) > 0

    def test_motion_response(self):
        """Test motion detection response."""
        eye = CompoundEye(eye_side="left")
        light_dir = np.array([0, 0, 1])

        # Sample with stationary light
        for _ in range(10):
            eye.sample_environment(light_dir, 1.0, np.array([1, 0, 0, 0]))

        # Motion response should be near zero
        motion_h, motion_v = eye.compute_motion()
        assert abs(motion_h) < 0.5
        assert abs(motion_v) < 0.5

    def test_phototaxis_signal(self):
        """Test phototaxis steering computation."""
        eye = CompoundEye(eye_side="left")
        light_dir = np.array([0, 0, 1])

        eye.sample_environment(light_dir, 1.0, np.array([1, 0, 0, 0]))
        yaw, pitch = eye.get_phototaxis_signal()

        # Should be finite values
        assert -1 <= yaw <= 1
        assert -1 <= pitch <= 1


class TestBinocularVision:
    """Test binocular vision system."""

    def test_binocular_creation(self):
        """Test binocular system initialization."""
        vision = BinocularVisionSystem(num_ommatidia_per_eye=100)
        assert vision.left_eye.num_ommatidia == 100
        assert vision.right_eye.num_ommatidia == 100

    def test_binocular_sampling(self):
        """Test binocular environment sampling."""
        vision = BinocularVisionSystem(num_ommatidia_per_eye=100)

        sample = vision.sample(
            light_direction=np.array([0, 0, 1]),
            light_intensity=1.0,
            body_orientation=np.array([1, 0, 0, 0]),
        )

        assert 'left_intensities' in sample
        assert 'right_intensities' in sample
        assert 'left_motion' in sample
        assert 'right_motion' in sample
        assert 'phototaxis' in sample

    def test_visual_encoder_output(self):
        """Test compression to neural input dimension."""
        vision = BinocularVisionSystem(num_ommatidia_per_eye=100)

        output = create_visual_encoder_output(vision, target_dim=64)
        assert len(output) == 64
        assert np.all(output >= 0)
        assert np.all(output <= 1)


class TestVisualIntegration:
    """Test integration with physics system."""

    def test_sensor_encoder_visual_dim(self):
        """Test SensorEncoder handles visual dimension."""
        encoder = SensorEncoder(
            num_proprioceptive=48,
            num_touch=6,
            num_visual=128,
        )
        assert encoder.output_dim == 48 + 6 + 128

    def test_encode_with_visual(self):
        """Test encoding with visual input."""
        from oneuro.physics.drosophila_simulator import DrosophilaPhysicsState, LegID

        encoder = SensorEncoder(num_proprioceptive=48, num_touch=6, num_visual=128)
        state = DrosophilaPhysicsState()

        # Set up joint positions/velocities for all 24 joints
        for joint in ['L1_C_pitch', 'L1_C_yaw', 'L1_F_pitch', 'L2_C_pitch', 'L2_C_yaw', 'L2_F_pitch',
                      'L3_C_pitch', 'L3_C_yaw', 'L3_F_pitch', 'R1_C_pitch', 'R1_C_yaw', 'R1_F_pitch',
                      'R2_C_pitch', 'R2_C_yaw', 'R2_F_pitch', 'R3_C_pitch', 'R3_C_yaw', 'R3_F_pitch',
                      'L1_T_pitch', 'L2_T_pitch', 'L3_T_pitch', 'R1_T_pitch', 'R2_T_pitch', 'R3_T_pitch']:
            state.joint_positions[joint] = 0.0
            state.joint_velocities[joint] = 0.0

        # Set up touch contacts
        touch_contacts = {leg: False for leg in LegID}

        visual_input = np.random.rand(128)
        pattern = encoder.encode(
            joint_positions=state.joint_positions,
            joint_velocities=state.joint_velocities,
            touch_contacts=touch_contacts,
            visual_input=visual_input,
        )

        assert len(pattern) == 182  # 48 + 6 + 128


# ============================================================
# Olfaction Tests
# ============================================================

from oneuro.physics.olfaction import (
    OdorantReceptor,
    Antenna,
    BilateralOlfaction,
    OdorantType,
    create_olfactory_encoder_output,
)


class TestOdorantReceptor:
    """Tests for single odorant receptor."""

    def test_receptor_creation(self):
        """Test receptor initialization."""
        receptor = OdorantReceptor(
            name="Or42a",
            sensitivity={"ethyl_acetate": 0.1},
        )
        assert receptor.name == "Or42a"
        assert "ethyl_acetate" in receptor.sensitivity

    def test_hill_response(self):
        """Test Hill function concentration-response."""
        receptor = OdorantReceptor(
            name="Or42a",
            sensitivity={"ethyl_acetate": 0.1},
            max_firing_rate=200.0,
            spontaneous_rate=5.0,
        )

        # Zero concentration -> spontaneous rate
        r0 = receptor.response(0.0, "ethyl_acetate")
        assert r0 == pytest.approx(5.0, rel=0.1)

        # EC50 concentration -> ~50% activation
        receptor.reset()
        r_ec50 = receptor.response(0.1, "ethyl_acetate")
        assert 50 < r_ec50 < 150  # Around half-max

        # High concentration -> near max
        receptor.reset()
        r_max = receptor.response(10.0, "ethyl_acetate")
        assert r_max > 150  # Near max firing

    def test_adaptation(self):
        """Test sensory adaptation over time."""
        receptor = OdorantReceptor(
            name="Or42a",
            sensitivity={"ethyl_acetate": 0.1},
            adaptation_tau=0.1,  # Fast adaptation
        )

        # Initial response
        r1 = receptor.response(1.0, "ethyl_acetate", dt=0.01)

        # Repeated stimulation -> adaptation
        for _ in range(50):
            receptor.response(1.0, "ethyl_acetate", dt=0.01)

        r_adapted = receptor.response(1.0, "ethyl_acetate", dt=0.01)

        # Adapted response should be lower
        assert r_adapted < r1

    def test_unknown_odorant(self):
        """Test response to odorant receptor doesn't sense."""
        receptor = OdorantReceptor(
            name="Or42a",
            sensitivity={"ethyl_acetate": 0.1},
        )
        r = receptor.response(1.0, "unknown_odor")
        assert r == pytest.approx(receptor.spontaneous_rate, rel=0.1)


class TestAntenna:
    """Tests for single antenna."""

    def test_antenna_creation(self):
        """Test antenna initialization."""
        antenna = Antenna(side="left", num_receptors=50)
        assert antenna.side == "left"
        assert len(antenna.receptors) == 50

    def test_sample_odorants(self):
        """Test odorant sampling."""
        antenna = Antenna(side="left", num_receptors=20)

        odorants = {"ethyl_acetate": 1.0}
        rates = antenna.sample(odorants)

        assert len(rates) == 20
        assert all(r > 0 for r in rates)  # All firing

    def test_chemotaxis_signal(self):
        """Test chemotaxis computation."""
        antenna = Antenna(side="left", num_receptors=20)

        # Attractive odor
        antenna.sample({"ethyl_acetate": 1.0})
        turn, approach = antenna.get_chemotaxis_signal()

        assert -1 <= turn <= 1
        assert -1 <= approach <= 1


class TestBilateralOlfaction:
    """Tests for bilateral olfactory system."""

    def test_bilateral_creation(self):
        """Test bilateral system initialization."""
        olfaction = BilateralOlfaction(num_receptors=50)
        assert olfaction.left_antenna.num_receptors == 50
        assert olfaction.right_antenna.num_receptors == 50

    def test_gradient_detection(self):
        """Test detection of odor gradient."""
        olfaction = BilateralOlfaction(num_receptors=20)

        # Uniform concentration
        sample1 = olfaction.sample({"ethyl_acetate": 1.0})
        turn1 = sample1['turn_signal']

        # With gradient (more on right)
        gradients = {"ethyl_acetate": np.array([0, 1.0, 0])}  # Gradient in +y
        sample2 = olfaction.sample({"ethyl_acetate": 1.0}, gradients)
        turn2 = sample2['turn_signal']

        # Turn signal should be positive (turn toward gradient)
        assert turn2 > turn1

    def test_olfactory_output(self):
        """Test olfactory encoder output."""
        olfaction = BilateralOlfaction(num_receptors=50)

        output = create_olfactory_encoder_output(
            olfaction,
            {"ethyl_acetate": 1.0},
            target_dim=100,
        )

        assert len(output) == 100
        # Note: turn_signal can be negative [-1, 1], so only check bounds
        assert np.all(output >= -1)
        assert np.all(output <= 1)

    def test_reset(self):
        """Test olfactory system reset."""
        olfaction = BilateralOlfaction(num_receptors=10)

        # Sample something
        olfaction.sample({"ethyl_acetate": 1.0})

        # Reset
        olfaction.reset()

        # Rates should be zero
        assert np.all(olfaction.left_antenna.current_rates == 0)
        assert np.all(olfaction.right_antenna.current_rates == 0)


# ============================================================
# Biological Parameters Tests
# ============================================================

from oneuro.physics.biological_params import (
    DrosophilaParameters,
    PARAMS,
    tune_mjcf_parameters,
    get_motor_pattern_parameters,
    get_sensory_parameters,
)


class TestBiologicalParameters:
    """Tests for biological parameter definitions."""

    def test_params_creation(self):
        """Test parameter initialization."""
        params = DrosophilaParameters()
        assert params.body_length > 0
        assert params.total_mass > 0
        assert params.SCALE == 1000.0

    def test_scaling(self):
        """Test value scaling for MuJoCo."""
        params = DrosophilaParameters()

        # Length scales linearly
        scaled = params.get_scaled_value(params.body_length)
        assert scaled == params.body_length * 1000

        # Mass scales with cube
        mass_scaled = params.get_scaled_mass(params.total_mass)
        assert mass_scaled == params.total_mass * (1000 ** 3)

    def test_mjcf_parameters(self):
        """Test MJCF parameter generation."""
        mjcf_params = tune_mjcf_parameters()

        assert 'timestep' in mjcf_params
        assert mjcf_params['timestep'] == 0.001
        assert 'thorax_mass' in mjcf_params
        assert mjcf_params['thorax_mass'] > 0

    def test_motor_pattern_params(self):
        """Test motor pattern parameters."""
        motor_params = get_motor_pattern_parameters()

        assert 'stride_frequency' in motor_params
        assert motor_params['stride_frequency'] > 0
        assert 'wing_frequency' in motor_params
        assert motor_params['wing_frequency'] == 200.0

    def test_sensory_params(self):
        """Test sensory parameters."""
        sensory = get_sensory_parameters()

        assert sensory['ommatidia_per_eye'] == 768
        assert sensory['receptor_types'] == 50
        assert sensory['max_firing_rate'] > 0

    def test_global_params(self):
        """Test global PARAMS singleton."""
        assert PARAMS.body_length == 2.5e-3
        assert PARAMS.wing_beat_frequency > 0
        assert len(PARAMS.leg_segments) == 4


# ============================================================
# GPU Batch Physics Tests
# ============================================================

class TestBatchPhysicsConfig:
    """Tests for batch physics configuration."""

    def test_config_creation(self):
        """Test configuration initialization."""
        from oneuro.physics.gpu_batch_physics import BatchPhysicsConfig
        config = BatchPhysicsConfig(num_flies=10)
        assert config.num_flies == 10
        assert config.timestep == 0.001
        assert config.arena_size == (8.0, 8.0)

    def test_config_custom_params(self):
        """Test custom configuration parameters."""
        from oneuro.physics.gpu_batch_physics import BatchPhysicsConfig
        config = BatchPhysicsConfig(
            num_flies=50,
            timestep=0.002,
            arena_size=(10.0, 10.0),
        )
        assert config.num_flies == 50
        assert config.timestep == 0.002
        assert config.arena_size == (10.0, 10.0)


class TestBatchPhysicsAvailability:
    """Tests for GPU batch physics availability checking."""

    def test_jax_availability_check(self):
        """Test JAX availability can be checked."""
        from oneuro.physics.gpu_batch_physics import JAX_AVAILABLE
        assert isinstance(JAX_AVAILABLE, bool)

    def test_mjx_availability_check(self):
        """Test MJX availability can be checked."""
        from oneuro.physics.gpu_batch_physics import MJX_AVAILABLE
        assert isinstance(MJX_AVAILABLE, bool)


class TestBatchEnvironmentCreation:
    """Tests for batch environment creation."""

    def test_create_batch_environment_imports(self):
        """Test that create_batch_environment can be imported."""
        from oneuro.physics.gpu_batch_physics import create_batch_environment
        assert callable(create_batch_environment)

    def test_batch_env_class_exists(self):
        """Test BatchDrosophilaEnv class exists."""
        from oneuro.physics.gpu_batch_physics import BatchDrosophilaEnv
        assert BatchDrosophilaEnv is not None
