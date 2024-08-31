import argparse

from failgen.env_wrapper import FailgenWrapper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task-name",
        type=str,
        default="PickCube-v1",
        help="The id of the task to be created for this demo",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Whether or not to open the GUI to visualize live",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Whether or not to save a video recording of the demo",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="The number of episodes to run this demo"
    )

    args = parser.parse_args()

    fail_wrapper = FailgenWrapper(
        task_name=args.task_name,
        headless=args.headless,
        save_video=args.save_video,
    )

    for i in range(args.num_episodes):
        _ = fail_wrapper.get_failure()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
