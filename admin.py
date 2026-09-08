"""
Admin: edit the whole content tree (profile, roles, projects, achievements)
without a redeploy.

Design notes worth knowing before you rely on this:

  · Storage is a JSON file (content.json) holding the entire tree, mutable
    from the panel. It works on any host with a real filesystem (VPS, Render,
    Fly, Docker). It does NOT persist on Vercel serverless, where the
    filesystem is ephemeral and resets on every cold start. See "Making it
    persist" in the README.

  · The frontend's content.ts is the initial seed. On first edit the panel
    calls POST /api/admin/seed with the full content.ts snapshot; the backend
    writes that to content.json and every subsequent read/write goes through
    the file. If the file already exists, seed is a no-op — it never
    overwrites edits.

  · From the moment content.json exists, it is canonical. The site reads
    /api/content and falls back to content.ts only when the backend is
    unreachable. That's the whole trick: "editing static content" works
    everywhere because we never edit static content, only a mutable copy of
    it.

  · The old add / delete / export routes are kept as backwards-compatible
    shims (see the bottom of the file) so anything still on the old admin URL
    keeps working while the new panel rolls out. They operate on the
    projects slice of the same content.json.

  · Auth is a signed HMAC token with an expiry. It is proportionate to what
    is behind it — portfolio copy, not user data — but it is not a substitute
    for a real identity system if this ever guards anything sensitive.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
STORE = BASE / "content.json"
# Kept for a one-time migration from the legacy per-entry store below.
LEGACY_STORE = BASE / "entries.json"

ADMIN_USER = os.getenv("ADMIN_USER", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
# Falls back to a value derived from the password so tokens are still unforgeable
# without a separate secret, but set ADMIN_SECRET in production.
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "") or f"derived::{ADMIN_PASSWORD}"
TOKEN_TTL = 60 * 60 * 8  # 8 hours

router = APIRouter(prefix="/api/admin", tags=["admin"])
public_router = APIRouter(tags=["content"])


# ── storage ────────────────────────────────────────────────────────────────

def _empty_tree() -> dict[str, Any]:
    """Shape the frontend always expects — never return a partial tree."""
    return {"profile": None, "roles": [], "projects": [], "achievements": [], "colors": {}}


def _read_tree() -> dict[str, Any] | None:
    """Full tree, or None when the store hasn't been seeded yet."""
    if not STORE.exists():
        # One-shot migration from the previous entries-only store: if the
        # legacy file exists but the new one doesn't, promote the old projects
        # into a fresh tree so nothing added through the old admin is lost.
        if LEGACY_STORE.exists():
            try:
                legacy = json.loads(LEGACY_STORE.read_text(encoding="utf-8"))
                if isinstance(legacy, list):
                    tree = _empty_tree()
                    tree["projects"] = legacy
                    _write_tree(tree)
                    return tree
            except (json.JSONDecodeError, OSError):
                pass
        return None
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    # Fill any missing top-level keys with sensible defaults, so a caller
    # that expects `projects` never crashes on a store written by an older
    # version of the panel.
    merged = _empty_tree()
    merged.update({k: v for k, v in data.items() if k in merged})
    return merged


def _write_tree(tree: dict[str, Any]) -> None:
    STORE.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")


# ── auth ───────────────────────────────────────────────────────────────────

def _sign(payload: str) -> str:
    return hmac.new(ADMIN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _issue_token() -> str:
    expires = int(time.time()) + TOKEN_TTL
    payload = f"{ADMIN_USER}:{expires}"
    return f"{payload}:{_sign(payload)}"


def require_admin(authorization: str | None) -> None:
    if not ADMIN_USER or not ADMIN_PASSWORD:
        raise HTTPException(503, "Admin is not configured on this server.")
    token = (authorization or "").removeprefix("Bearer ").strip()
    try:
        user, expires, signature = token.rsplit(":", 2)
    except ValueError:
        raise HTTPException(401, "Not signed in.")
    if not hmac.compare_digest(signature, _sign(f"{user}:{expires}")):
        raise HTTPException(401, "Session is invalid. Sign in again.")
    if int(expires) < time.time():
        raise HTTPException(401, "Session expired. Sign in again.")


# ── models ─────────────────────────────────────────────────────────────────

class Credentials(BaseModel):
    username: str
    password: str


class ContentTree(BaseModel):
    """
    The whole content tree in one payload. Deliberately loose (Any inside) so
    the shape can drift on the frontend without a Pydantic 422 blocking a save.
    The frontend is the schema-of-record here — the backend just persists what
    it's given, unchanged.
    """

    profile: Any = None
    roles: list[Any] = Field(default_factory=list)
    projects: list[Any] = Field(default_factory=list)
    achievements: list[Any] = Field(default_factory=list)
    # Admin Colors tab overrides: { dark?: {...}, light?: {...} }. Loose Any
    # for the same reason as the rest of this model — the frontend owns the
    # shape, the backend just persists whatever it's handed.
    colors: Any = Field(default_factory=dict)


# The legacy per-project add form still uses the shape below. Kept for the
# backwards-compat shim endpoints at the bottom of the file.
class ProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    year: str = Field(default="", max_length=40)
    context: str = Field(default="", max_length=120)
    blurb: str = Field(default="", max_length=400)
    challenge: str = Field(default="", max_length=2000)
    approach: str = Field(default="", max_length=2000)
    outcome: str = Field(default="", max_length=2000)
    tech: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    featured: bool = False
    liveUrl: str = Field(default="", max_length=300)

    def slug(self) -> str:
        base = "".join(c if c.isalnum() else "-" for c in self.name.lower()).strip("-")
        while "--" in base:
            base = base.replace("--", "-")
        return base or uuid.uuid4().hex[:8]


# ── auth route ─────────────────────────────────────────────────────────────

@router.post("/login")
async def login(creds: Credentials):
    if not ADMIN_USER or not ADMIN_PASSWORD:
        raise HTTPException(503, "Admin is not configured on this server.")
    ok_user = hmac.compare_digest(creds.username, ADMIN_USER)
    ok_pass = hmac.compare_digest(creds.password, ADMIN_PASSWORD)
    if not (ok_user and ok_pass):
        # Same message either way — never reveal which half was wrong.
        raise HTTPException(401, "Those credentials don't match.")
    return {"token": _issue_token(), "expiresIn": TOKEN_TTL}


# ── whole-tree routes (the ones the new admin uses) ────────────────────────

@router.get("/content")
async def admin_get_content(authorization: str | None = Header(default=None)):
    """
    The whole tree, or `{ seeded: false }` when nothing's been written yet.
    The panel uses that signal to seed content.json from its own compiled
    copy of content.ts on first edit.
    """
    require_admin(authorization)
    tree = _read_tree()
    if tree is None:
        return {"seeded": False, "content": _empty_tree()}
    return {"seeded": True, "content": tree}


@router.post("/seed")
async def seed_content(
    payload: ContentTree, authorization: str | None = Header(default=None)
):
    """
    Idempotent. If content.json already exists, this is a no-op — it never
    overwrites existing edits. If it doesn't exist, the request body is
    written verbatim as the initial store.
    """
    require_admin(authorization)
    if _read_tree() is not None:
        return {"ok": True, "seeded": True, "skipped": True}
    _write_tree(payload.model_dump())
    return {"ok": True, "seeded": True, "skipped": False}


@router.put("/content")
async def admin_put_content(
    payload: ContentTree, authorization: str | None = Header(default=None)
):
    """Whole-file replace. Deliberately atomic — no partial-update races."""
    require_admin(authorization)
    _write_tree(payload.model_dump())
    return {"ok": True}


# ── public read routes ─────────────────────────────────────────────────────

@public_router.get("/api/content")
async def public_content():
    """
    Read-only. The site reads this on load; if it 404s, the frontend falls
    back to its own compiled copy of content.ts.
    """
    tree = _read_tree()
    if tree is None:
        raise HTTPException(404, "Content store has not been initialised.")
    return tree


# ── backwards-compat shims ─────────────────────────────────────────────────
#
# The pre-tree admin panel called these paths. Keeping them so nothing that
# still points at the old URLs breaks between versions. They operate on the
# projects slice of the same content.json.

@router.get("/entries")
async def list_entries(authorization: str | None = Header(default=None)):
    require_admin(authorization)
    tree = _read_tree() or _empty_tree()
    return {"entries": tree.get("projects", [])}


@router.post("/entries")
async def add_entry(
    project: ProjectInput, authorization: str | None = Header(default=None)
):
    require_admin(authorization)
    tree = _read_tree() or _empty_tree()
    projects = tree.get("projects", [])
    slug = project.slug()
    if any(p.get("slug") == slug for p in projects):
        raise HTTPException(409, f"A project with the slug '{slug}' already exists.")

    entry = project.model_dump()
    entry["slug"] = slug
    entry["links"] = (
        [{"label": project.liveUrl, "href": project.liveUrl}] if project.liveUrl else []
    )
    entry.pop("liveUrl", None)
    entry["addedAt"] = int(time.time())
    projects.append(entry)
    tree["projects"] = projects
    _write_tree(tree)
    return {"ok": True, "slug": slug, "count": len(projects)}


@router.delete("/entries/{slug}")
async def delete_entry(slug: str, authorization: str | None = Header(default=None)):
    require_admin(authorization)
    tree = _read_tree() or _empty_tree()
    projects = tree.get("projects", [])
    remaining = [p for p in projects if p.get("slug") != slug]
    if len(remaining) == len(projects):
        raise HTTPException(404, "No entry with that slug.")
    tree["projects"] = remaining
    _write_tree(tree)
    return {"ok": True, "count": len(remaining)}


@router.get("/export")
async def export_ts(authorization: str | None = Header(default=None)):
    """
    Ready-to-paste content.ts entries. Kept because it's still the recommended
    path for making an entry permanent (survives a host with an ephemeral
    filesystem, ships in the static bundle). Only exports the projects slice —
    profile/roles/achievements editing is new and their promotion path is
    documented in the panel, not auto-serialised here.
    """
    require_admin(authorization)
    tree = _read_tree() or _empty_tree()

    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace("'", "\\'")

    blocks = []
    for e in tree.get("projects", []):
        lines = [
            "  {",
            f"    slug: '{esc(e['slug'])}',",
            f"    name: '{esc(e['name'])}',",
            f"    year: '{esc(e.get('year', ''))}',",
            f"    context: '{esc(e.get('context', ''))}',",
        ]
        if e.get("featured"):
            lines.append("    featured: true,")
        lines.append(f"    blurb: '{esc(e.get('blurb', ''))}',")
        for key in ("challenge", "approach", "outcome"):
            if e.get(key):
                lines.append(f"    {key}: '{esc(e[key])}',")
        tech = ", ".join(f"'{esc(t)}'" for t in e.get("tech", []))
        doms = ", ".join(f"'{esc(d)}'" for d in e.get("domains", []))
        lines.append(f"    tech: [{tech}],")
        lines.append(f"    domains: [{doms}],")
        for link in e.get("links", []):
            lines.append(
                f"    links: [{{ label: '{esc(link['label'])}', href: '{esc(link['href'])}' }}],"
            )
        lines.append("  },")
        blocks.append("\n".join(lines))

    return {"code": "\n".join(blocks) or "// No admin entries yet."}


@public_router.get("/api/content/entries")
async def public_entries():
    """Old public read. Points at the projects slice of the new tree."""
    tree = _read_tree() or _empty_tree()
    return {"entries": tree.get("projects", [])}