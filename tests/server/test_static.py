"""Production frontend hosting must preserve API and WebSocket routes."""

from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app


def test_api_starts_without_frontend_or_map_downloads(tmp_path: Path) -> None:
    (tmp_path / "demo_allowlist.json").write_text("[]")
    with TestClient(create_app(maps_dir=tmp_path)) as client:
        assert client.get("/api/games").json() == [{"id": "saber", "label": "Fly Saber"}]
        assert client.get("/api/maps").json() == []
        assert client.get("/").status_code == 404


def test_frontend_mount_preserves_api_and_websocket(tmp_path: Path) -> None:
    maps = tmp_path / "maps"
    maps.mkdir()
    (maps / "demo_allowlist.json").write_text("[]")
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text("<!doctype html><title>Fly Saber</title>")
    (frontend / "app.js").write_text("console.log('Fly Saber')")
    with TestClient(create_app(maps_dir=maps, frontend_dir=frontend)) as client:
        homepage = client.get("/")
        assert homepage.status_code == 200
        assert "text/html" in homepage.headers["content-type"]
        assert "Fly Saber" in homepage.text
        assert client.get("/app.js").text == "console.log('Fly Saber')"
        assert client.get("/api/games").json() == [{"id": "saber", "label": "Fly Saber"}]
        assert client.get("/api/maps/missing_easy").status_code == 404
        assert client.get("/missing.js").status_code == 404
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({"type": "warp"})
            assert ws.receive_json()["type"] == "error"
