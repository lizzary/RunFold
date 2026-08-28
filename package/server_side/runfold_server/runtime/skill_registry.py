from __future__ import annotations

import re
from pathlib import Path

import yaml

from runfold_server.errors import StartupError
from runfold_server.runtime.models import Skill

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SELECTOR = re.compile(r"(?<![A-Za-z0-9_/])/(?P<name>[a-z0-9][a-z0-9-]{0,63})\b")


class SkillRegistry:
    def __init__(self, root: Path) -> None:
        self._skills = _load_skills(root)

    def catalog(self) -> tuple[Skill, ...]:
        return tuple(self._skills[name] for name in sorted(self._skills))

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def select_for_task(self, task: str) -> tuple[str, tuple[Skill, ...]]:
        selected_names: list[str] = []

        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            if name not in self._skills:
                return match.group(0)
            if name not in selected_names:
                selected_names.append(name)
            return ""

        cleaned = " ".join(_SELECTOR.sub(replace, task).split())
        return cleaned, tuple(self._skills[name] for name in selected_names)


def _load_skills(root: Path) -> dict[str, Skill]:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise StartupError("invalid_skills", "Agent skill directory is unavailable") from error
    if not resolved_root.is_dir():
        raise StartupError("invalid_skills", "Agent skill directory is unavailable")

    skills: dict[str, Skill] = {}
    for path in sorted(resolved_root.glob("*/SKILL.md")):
        resolved_path = path.resolve(strict=True)
        if not resolved_path.is_relative_to(resolved_root):
            raise StartupError("invalid_skills", "Agent skill path escapes the skill directory")
        skill = _read_skill(resolved_path)
        if skill.name in skills:
            raise StartupError("invalid_skills", "Agent skill names must be unique")
        skills[skill.name] = skill
    if not skills:
        raise StartupError("invalid_skills", "At least one agent skill is required")
    return skills


def _read_skill(path: Path) -> Skill:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise StartupError("invalid_skills", "Agent skill cannot be read") from error
    if not content.startswith("---\n"):
        raise StartupError("invalid_skills", "Agent skill format is invalid")
    frontmatter, separator, body = content[4:].partition("\n---\n")
    if not separator:
        raise StartupError("invalid_skills", "Agent skill format is invalid")
    try:
        metadata = yaml.safe_load(frontmatter)
    except yaml.YAMLError as error:
        raise StartupError("invalid_skills", "Agent skill metadata is invalid") from error
    if not isinstance(metadata, dict) or set(metadata) != {"name", "description"}:
        raise StartupError("invalid_skills", "Agent skill metadata is invalid")
    name = metadata.get("name")
    description = metadata.get("description")
    if (
        not isinstance(name, str)
        or _NAME.fullmatch(name) is None
        or path.parent.name != name
        or not isinstance(description, str)
        or not description.strip()
        or not body.strip()
    ):
        raise StartupError("invalid_skills", "Agent skill metadata is invalid")
    return Skill(
        name=name,
        description=description.strip(),
        instructions=body.strip(),
    )
