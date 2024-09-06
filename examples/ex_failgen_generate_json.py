import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT_FOLDER = "/home/gregor/data/failgen_data/"
DEFAULT_OUTPUT_FOLDER = "/home/gregor/data/failgen_data/"
FOLDER_NAME_PATTERN = r"^(\d+)_([a-zA-Z_]+)_(\d+)$"

STR_HUMAN_PROMPT ="""In the image, it contains many frames from different views
at different time steps in a matrix arrangement depicting the robot arm
performing push the block. The frames are combined with different views, from
top to bottom: the top row is the front view; the middle row is the wrist view,
and the bottom is the overhead view. Each frame is labeled with the
corresponding timestep reflecting the temporal information from left to right.
The white grid represents future sub-tasks that has yet to happen and should be
ignored. For the given sub-tasks, first determine it has succeeded by choosing
from [\"yes\", \"no\"], and if it is \"no\" then explain the reason why the
current sub-tasks has failed. For example: \"No, The robot slip the object out
of its gripper.\"
"""

FAILURE_REASONS = {
    "grasp": "The robot gripper fails to close the gripper.",
    "trans_x": "The robot gripper move to the desired position with"
    + " an offset along the x direction.",
    "trans_y": "The robot gripper move to the desired position with"
    + " an offset along the y direction.",
    "trans_z": "The robot gripper move to the desired position with"
    + " an offset along the z direction.",
}


def collect_from_task(
    task_name: str, input_folder: Path, fail_entries: List[Any]
) -> None:
    path_to_task = (input_folder / task_name).resolve()
    if not path_to_task.exists():
        return
    if not path_to_task.is_dir():
        return

    episodes_data = []
    for candidate_fold in path_to_task.iterdir():
        if not candidate_fold.is_dir():
            continue
        match_re = re.match(FOLDER_NAME_PATTERN, candidate_fold.name)
        if match_re:
            ep_parsed = (
                candidate_fold,
                int(match_re.group(1)),
                match_re.group(2),
                int(match_re.group(3)),
            )
            episodes_data.append(ep_parsed)

    idx = 0
    for fail_dir, _, fail_type, _ in episodes_data:
        png_files = list(fail_dir.glob("*.png"))
        png_files = sorted(png_files, key=lambda x: int(x.stem))
        for png_file in png_files:
            fail_dict = {
                "id": idx + len(fail_entries),
                "image": str(png_file),
                "conversations": [
                    {
                        "from": "human",
                        "value": STR_HUMAN_PROMPT,
                    },
                    {
                        "from": "gpt",
                        "value": FAILURE_REASONS[fail_type],
                    },
                ],
            }
            fail_entries.append(fail_dict)
            idx += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder",
        type=str,
        default=DEFAULT_INPUT_FOLDER,
        help="The folder where to look for the generated images",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help="The folder where to place the generated json file",
    )

    args = parser.parse_args()

    TASKS = [
        "FailPickCube-v1",
        "FailPushCube-v1",
        "FailPlugCharger-v1",
        "FailStackCube-v1",
        "FailPegInsertionSide-v1",
    ]

    fail_entries = []
    for task in TASKS:
        collect_from_task(task, Path(args.input_folder), fail_entries)

    json_filepath = (Path(args.output_folder) / "outman_qa.json").resolve()
    with open(json_filepath, "w") as fhandle:
        json.dump(fail_entries, fhandle, indent=4)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
