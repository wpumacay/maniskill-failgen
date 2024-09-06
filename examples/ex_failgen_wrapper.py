import argparse

import numpy as np
from failgen.env_wrapper import FailgenWrapper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task-name",
        type=str,
        default="FailPickCube-v1",
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
        help="The number of episodes to run this demo",
    )

    args = parser.parse_args()

    fail_wrapper = FailgenWrapper(
        task_name=args.task_name,
        headless=args.headless,
        save_video=args.save_video,
    )

    collected = 0
    for i in range(args.num_episodes):
        stage_idx = np.random.choice(fail_wrapper._fail_plan_wrapper.stages)
        fail_wrapper._fail_plan_wrapper.set_active_stage(stage_idx)
        success = fail_wrapper.get_failure()
        print(f"{i+1}/{args.num_episodes} - Success: {success}")
        if not success:
            fail_wrapper.save_video(save=True, ep_idx=collected)
            collected += 1
        else:
            fail_wrapper.save_video(save=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
