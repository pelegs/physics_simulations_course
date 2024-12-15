from lib.AABB import AABB
from lib.constants import npdarr


class Object:
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

    def move_to(self, new_pos: npdarr) -> None:
        self.translate(new_pos - self.pos)

    def translate(self, dr: npdarr) -> None:
        self.pos = self.pos + dr
        self.bbox.translate(dr)


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
        self.translate(self.vel * dt)
