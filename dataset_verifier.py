from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Replace with your dataset repo ID or local path
dataset = LeRobotDataset("robotnana/screw_picking")
print(f"Total episodes: {dataset.meta['total_episodes']}")
# This should show 3 if you did three recordings.
