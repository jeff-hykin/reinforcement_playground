import carla_env

carla_env.register(
    id='CarlaEnv-state-town01-v1',
    entry_point='carla_env.carla_env:CarlaEnv',
    max_episode_steps=500,
    kwargs={
        'render': True,
        'carla_port': 2000,
        'changing_weather_speed': 0.1,
        'frame_skip': 1,
        'observations_type': 'state',
        'traffic': True,
        'vehicle_name': 'tesla.cybertruck',
        'map_name': 'Town01',
        'autopilot': True
    }
)


