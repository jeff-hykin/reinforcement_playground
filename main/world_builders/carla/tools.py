import random
import time
import carla

class Carla:
    """
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
    """
    def __init__(self, address='localhost', port=2000, settings=dict()):
        self.client = carla.Client(address, port)
        self.world = client.get_world()
        # set the settings
        self.settings = world.get_settings()
        for each_key, each_value in settings.items():
            setattr(self.settings, each_key, each_value)
        self.world.apply_settings(self.settings)
        self.vehicles = []
        self.cameras = []
    
    @property
    def blueprints(self):
        return world.get_blueprint_library()
    
    def spawn_points(self):
        return self.world.get_map().get_spawn_points()
    
    def spawn_camera(self, location, rotation, attach_to, height=720, width=1280,):
        # blueprint
        rgb_camera_blueprint = self.blueprints.find('sensor.camera.rgb')
        rgb_camera_blueprint.set_attribute('image_size_y', f'{height}')
        rgb_camera_blueprint.set_attribute('image_size_x', f'{width}')
        # actual camera
        camera = self.world.spawn_actor(
            rgb_camera_blueprint,
            carla.Transform(location=location, rotation=rotation),
            attach_to=attach_to,
        )
        self.cameras.append(camera)
        return camera
        
    def spawn_vehicle(self, attributes=dict(role_name="autopilot"), blueprint=None, number_of_wheels=4, spawn_point=None):
        """ This function spawns the driving vehicle and puts it into an autopilot mode.
        Returns:
            A carla.Actor instance representing the vehicle that was just spawned.
        """
        # 
        # pick blueprint
        # 
        if blueprint:
            vehicle_blueprint = blueprint
        else: # pick randomly
            vehicle_blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.*')) 
            while (
                    not vehicle_blueprint.has_attribute('number_of_wheels')
                    or
                    not int(vehicle_blueprint.get_attribute('number_of_wheels')) == number_of_wheels
                ):
                vehicle_blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
        
        # 
        # set attributes
        # 
        attributes = { "role_name": "autopilot", **attributes } # apply the default value
        for each_key, each_value in attributes.items():
            vehicle_blueprint.set_attribute(each_key, each_value)
        
        # 
        # set spawn point
        # 
        start_pose = spawn_point if spawn_point is not None else random.choice(self.world.get_map().get_spawn_points())
        
        # 
        # create vehicle
        # 
        command_result = carla.command.SpawnActor(vehicle_blueprint, start_pose)
        if attributes["role_name"] == "autopilot":
            command_result.then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)
            )
        vehicle_id = self.client.apply_batch_sync([batch])[0].actor_id
        # FIXME: there should be a better way to check if the vehicle is finished spawning
        time.sleep(0.5)
        self.vehicles.append(self.world.get_actors().find(vehicle_id))
        return self.vehicles[-1]
    
    def destroy(self):
        # vehicles
        client.apply_batch([carla.command.DestroyActor(each_vehicle) for each_vehicle in self.vehicles])
        # cameras
        client.apply_batch([carla.command.DestroyActor(each_camera) for each_camera in self.cameras])