from dataclasses import dataclass, field


@dataclass
class ConfluencePage:
    page_id: str
    title: str
    parent_id: str | None = None
    level: int = 0
    children: list["ConfluencePage"] = field(default_factory=list)