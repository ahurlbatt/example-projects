def run_test():

    import matplotlib.pyplot as plt
    import pathlib

    from flirseq import seq_reader
    from flirseq import infroptics

    # Find out where we are
    my_dir = pathlib.Path(__file__).parent.resolve()

    # The name of the file used for testing
    file_name = str(my_dir) + "/SC655.seq"

    # Define the environment conditions
    background_temperature = 20 + 273.15
    relative_humidity = 0.5

    # Create a list of optical components between the camera and the object
    my_optics = [
        infroptics.Atmosphere(background_temperature, 0.1, relative_humidity),
        infroptics.Window(background_temperature, 0.97, 0.01, 0.01),
        infroptics.Vacuum(0.3),
        infroptics.Mirror(background_temperature, 0.99),
        infroptics.Vacuum(1),
    ]

    # Define the estimated emittance of the object of interest
    object_emittance = 0.9

    # Pass these arguments to the top-level function for direct conversion to Kelvin of the last frame in the file
    K, s = seq_reader.seq_to_kelvin(file_name, my_optics, background_temperature, object_emittance, frames="last")

    # Extract the times of the frames that have been read
    frame_times = s.frame_times

    # Show the last frame as an image in degrees Centigrade
    imgplt = plt.imshow(K[0] - 273.15)

    return s, frame_times, imgplt, my_optics


if __name__ == "__main__":
    s, frame_times, imgplt, my_optics = run_test()
