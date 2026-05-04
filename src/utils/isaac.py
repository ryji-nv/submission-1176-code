import os


def needs_render(images_dir: str) -> bool:
    return not (os.path.isdir(images_dir) and os.listdir(images_dir))


def launch_sim():
    """Launch Isaac Sim headless, suppressing its startup output."""
    print("Launching Isaac Sim ...", flush=True)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        from isaacsim import SimulationApp

        app = SimulationApp(launch_config={"headless": True})
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])
    print("Isaac Sim ready.", flush=True)
    return app
