"""Empty table + dual UR5 playground for the interactive tutorial.

Same table, wall, cameras, and arm setup as the base suite. No task props.
"""
from ._base_task import Base_Task


class tutorial_empty(Base_Task):
    """Bare tabletop with both arms at home. Used by tutorial parts 1–4."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        return

    def play_once(self):
        pass

    def check_success(self):
        return False
