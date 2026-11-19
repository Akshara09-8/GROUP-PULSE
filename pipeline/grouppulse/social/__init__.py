"""Phase 2: group dynamics from per-frame gaze targets and torso direction.

Input: the track table from grouppulse.perception.run (one row per person per sampled frame).
Output: per-person scores (attention received, dominance, engagement, disengaged spells), group
scores (mutual gaze, gaze following, facing, rapport), a per-window timeline and a robot-addressee
suggestion per window.
"""

from .metrics import Series, SocialConfig, analyze, load_social_config

__all__ = ["Series", "SocialConfig", "analyze", "load_social_config"]
