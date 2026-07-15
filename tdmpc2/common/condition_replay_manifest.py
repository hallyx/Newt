from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def condition_id(assembly_id: Any, template_id: Any = None, fallback_template: str = "default") -> str:
	assembly = str(assembly_id).strip()
	template = str(template_id if template_id is not None else fallback_template).strip()
	return f"{assembly}|{template or fallback_template}"


def load_manifest(path):
	path = Path(path).expanduser()
	if not path.exists():
		return {
			"version": 1,
			"kind": "condition_replay_manifest",
			"conditions": [],
		}
	with open(path, "r", encoding="utf-8") as f:
		payload = json.load(f)
	if "conditions" not in payload:
		payload["conditions"] = payload.get("tasks", [])
	return payload


def write_manifest(path, payload):
	path = Path(path).expanduser()
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=True, indent=2)
		f.write("\n")
	return path


def upsert_condition(payload, entry):
	conditions = payload.setdefault("conditions", payload.get("tasks", []))
	cid = str(entry.get("condition_id") or condition_id(entry.get("assembly_id"), entry.get("template_id")))
	entry = dict(entry)
	entry["condition_id"] = cid
	conditions = [item for item in conditions if str(item.get("condition_id")) != cid]
	conditions.append(entry)
	conditions.sort(key=lambda item: str(item.get("condition_id", "")))
	payload["conditions"] = conditions
	return payload
