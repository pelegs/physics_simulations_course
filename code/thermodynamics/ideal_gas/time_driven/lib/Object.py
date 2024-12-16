from lib.AABB import AABB
from lib.constants import npdarr


class Object:
    """Docstring for Object."""

    id: int = -1

    def __init__(
        self,
        pos: npdarr,
        bbox_pts: npdarr,
        color: str = "#FF0000",
        opacity: float = 1.0,
    ) -> None:
        self.pos: npdarr = pos
        self.bbox = AABB(obj=self, pts=bbox_pts)
        self.color = color
        self.opacity = opacity

    def set_pos(self, new_pos: npdarr) -> None:
        self.pos = new_pos
        self.bbox.set_pos(new_pos)


class MovingObject(Object):
    """Docstring for MovingObject."""

    def __init__(
        self,
        pos: npdarr,
        vel: npdarr,
        bbox_pts: npdarr,
        color: str = "#FF0000",
        opacity: float = 1.0,
    ):
        super().__init__(pos, bbox_pts, color, opacity)
        self.vel: npdarr = vel

    def move(self, dt: float):
        self.set_pos(self.pos + self.vel * dt)
