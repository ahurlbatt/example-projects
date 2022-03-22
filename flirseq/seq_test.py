def run_test():

    import matplotlib.pyplot as plt
    import pathlib

    from flirseq import seq_reader
    from flirseq import ir_imaging

    # Find out where we are
    my_dir = pathlib.Path(__file__).parent.resolve()

    # The name of the file used for testing
    file_name = str(my_dir) + "/SC655.seq"

    # Define the environment conditions
    background_temperature = 20 + 273.15
    relative_humidity = 0.5

    # Create a list of optical components between the camera and the object
    object_system = [
        ir_imaging.Atmosphere(temperature=background_temperature, length=0.1, relative_humidity=relative_humidity),
        ir_imaging.Window(
            temperature=background_temperature,
            background_temperature=background_temperature,
            transmittance=0.97,
            reflectance=0.02,
            axis_alignment=True,
        ),
        ir_imaging.Vacuum(0.3),
        ir_imaging.Mirror(
            reflectance=0.99, temperature=background_temperature, background_temperature=background_temperature,
        ),
        ir_imaging.Vacuum(1),
    ]

    # Define the estimated emittance of the object of interest
    target_object = ir_imaging.TargetObject(
        emittance=0.9, background_temperature=background_temperature, diffuse_fraction=1.0,
    )

    # Pass these arguments to the top-level function for direct conversion to Kelvin of the last frame in the file
    K, s = seq_reader.seq_to_kelvin(file_name, target_object, object_system, frames="last")

    # Extract the times of the frames that have been read
    frame_times = s.frame_times

    # Show the last frame as an image in degrees Centigrade
    imgplt = plt.imshow(K[0] - 273.15)

    return s, frame_times, imgplt, object_system


if __name__ == "__main__":
    s, frame_times, imgplt, object_system = run_test()
