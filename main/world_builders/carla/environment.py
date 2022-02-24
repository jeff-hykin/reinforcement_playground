from main.world_builders.carla.tools import Carla, carla

def main():
    carla_instance = Carla(
        settings=dict(
            synchronous_mode=True,
            fixed_delta_seconds=1.0/10,
        ),
    )

    # Spawn the vehicle.
    vehicle = carla_instance.spawn_vehicle()
    # location = vehicle.get_location()
    # location.x += 40
    # vehicle.set_location(location)

    # Spawn the camera and register a function to listen to the images.
    camera = carla_instance.spawn_camera(
        location=carla.Location(x=2.0, y=0.0, z=1.8),
        rotation=carla.Rotation(roll=0, pitch=0, yaw=0),
        attach_to=vehicle,
    )
    camera.listen(
        lambda image:
            image.save_to_disk(
                f'carla.output.ignore/{image.frame:06d}.png',
                carla.ColorConverter.LogarithmicDepth,
            )
    )
    
    carla_instance.world.tick()