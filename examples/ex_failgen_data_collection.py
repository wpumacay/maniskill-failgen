import argparse

from failgen.env_wrapper import FailgenWrapper


def run_get_failures(
    task_name: str,
    fail_type: str,
    num_episodes: int,
    save_video: bool,
    headless: bool,
    max_tries: int = 10,
) -> None:
    fail_wrapper = FailgenWrapper(
        task_name=task_name,
        headless=headless,
        save_video=save_video,
    )

    fail_wrapper._fail_plan_wrapper.set_active_type(fail_type)
    fail_obj = fail_wrapper._fail_plan_wrapper._active_fail

    if fail_obj is None:
        return

    stages = fail_obj.stages

    collected_episodes = 0
    for stage in stages:
        curr_tries = max_tries
        fail_wrapper._fail_plan_wrapper.set_active_stage(stage)
        while collected_episodes <= num_episodes and curr_tries > 0:
            success = fail_wrapper.get_failure()
            print(
                f"stage: {stage}, success: {success}, fail_type: {fail_type}, num_ep: {collected_episodes}, curr_tries: {curr_tries}"
            )
            if not success:
                curr_tries = max_tries
                collected_episodes += 1
                fail_wrapper.save_video(save=True, ep_idx=collected_episodes)
            else:
                fail_wrapper.save_video(save=False)
                curr_tries -= 1


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
    parser.add_argument(
        "--fail-type",
        type=str,
        default="",
        help="The specific fail type to use for data collection",
    )

    args = parser.parse_args()

    FAIL_TYPES = [
        "grasp",
        "trans_x",
        "trans_y",
        "trans_z",
    ]

    if args.fail_type != "":
        FAIL_TYPES = [args.fail_type]

    for fail_type in FAIL_TYPES:
        run_get_failures(
            task_name=args.task_name,
            fail_type=fail_type,
            num_episodes=args.num_episodes,
            save_video=args.save_video,
            headless=args.headless,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
