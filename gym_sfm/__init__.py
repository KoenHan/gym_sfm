from gym.envs.registration import register

register(
    id = 'gym_sfm-v0',
    entry_point = 'gym_sfm.envs:GymSFM'
)
