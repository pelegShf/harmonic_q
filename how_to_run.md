python run.py --env VelocityGrid-v0 --variants both --seeds 0-4 \
  --episodes 8000 --eps_decay_episodes 7600 --smooth 301 \
  --task all --log_root logs/vel8k

# MultiStepGrid — 8k episodes
python run.py --env MultiStepGrid-v0 --variants both --seeds 0-4 \
  --episodes 8000 --eps_decay_episodes 7600 --smooth 301 \
  --task all --log_root logs/multi8k


python run.py --env VelocityGrid-v0  --task compare --log_root logs/vel8k  --smooth 301 --tag VelocityGrid_8k
python run.py --env MultiStepGrid-v0 --task compare --log_root logs/multi8k --smooth 301 --tag MultiStepGrid_8k
