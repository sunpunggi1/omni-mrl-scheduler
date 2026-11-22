"""
순수 Python 표준 라이브러리(`ThreadingHTTPServer`) 기반 Kubernetes `kube-scheduler` HTTP Extender 서버.

제공 엔드포인트:
- `GET  /healthz`              : Liveness/Readiness 헬스체크
- `GET  /metrics`              : 온라인 게이트(rho, alpha, cv, gamma) 및 레이턴시 메트릭 (JSON / `?format=prometheus`)
- `POST /filter`               : 노드 필터링 (`ExtenderArgs` -> `ExtenderFilterResult`)
- `POST /prioritize`           : 노드 스코어링 (`ExtenderArgs` -> `HostPriorityList`)
- `POST /gang_schedule`        : 멀티노드 Rack-Aware Gang 배치 확장
- `POST /events/pod_completed` : 완료 파드 수명 이벤트 수신 (`(gpu, duration)`)
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from k8s_extender.adapter import OmniLindyK8sAdapter

# 악의적·비정상 초대형 페이로드 방어 한도 (16 MiB)
_MAX_CONTENT_LENGTH = 16 * 1024 * 1024


class K8sExtenderServer(ThreadingHTTPServer):
    """어댑터 인스턴스와 로깅 설정을 보관하는 멀티스레드 K8s HTTP Extender 서버."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: type,
        adapter: OmniLindyK8sAdapter,
        quiet: bool = False,
    ) -> None:
        self.adapter = adapter
        self.quiet = quiet
        super().__init__(server_address, RequestHandlerClass)


class _ExtenderRequestHandler(BaseHTTPRequestHandler):
    """K8s Scheduler Extender HTTP/1.1 요청 핸들러."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        server = self.server
        if getattr(server, "quiet", False):
            return
        super().log_message(format, *args)

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", raw)

    def _read_json_body(self) -> Dict[str, Any]:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            raise ValueError("Content-Length 헤더가 필요합니다.")
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ValueError(f"유효하지 않은 Content-Length입니다: {raw_len!r}") from exc
        if length < 0 or length > _MAX_CONTENT_LENGTH:
            raise ValueError(f"요청 본문 크기({length} bytes)가 허용 범위를 초과했습니다.")
        raw_bytes = self.rfile.read(length)
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"유효하지 않은 JSON 본문입니다: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("요청 본문 최상위는 JSON object여야 합니다.")
        return data

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        adapter: OmniLindyK8sAdapter = self.server.adapter  # type: ignore[attr-defined]

        if parsed.path == "/healthz":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "scheduler": "Omni-Lindy",
                    "engine": adapter.engine,
                    "version": "11.1" if adapter.engine == "omni_mrl" else "7.0",
                },
            )
            return

        if parsed.path == "/metrics":
            qs = parse_qs(parsed.query)
            if qs.get("format", [""])[0].lower() == "prometheus":
                body = adapter.format_prometheus_metrics().encode("utf-8")
                self._send_bytes(200, "text/plain; version=0.0.4; charset=utf-8", body)
            else:
                self._send_json(200, adapter.get_metrics())
            return

        self._send_json(404, {"error": f"지원하지 않는 경로입니다: {parsed.path}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        adapter: OmniLindyK8sAdapter = self.server.adapter  # type: ignore[attr-defined]

        if parsed.path not in ("/filter", "/prioritize", "/gang_schedule", "/events/pod_completed"):
            self._send_json(404, {"error": f"지원하지 않는 경로입니다: {parsed.path}"})
            return

        try:
            body = self._read_json_body()
            if parsed.path == "/filter":
                res = adapter.filter_nodes(body)
            elif parsed.path == "/prioritize":
                res = adapter.prioritize_nodes(body)
            elif parsed.path == "/gang_schedule":
                res = adapter.schedule_gang(body)
            else:
                res = adapter.record_completions(body)
            self._send_json(200, res)
        except (ValueError, KeyError, TypeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - 예기치 못한 내부 오류 방어
            self._send_json(500, {"error": f"내부 스케줄러 오류: {exc}"})


def create_extender_server(
    host: str = "127.0.0.1",
    port: int = 8888,
    nodes_per_rack: int = 8,
    quiet: bool = False,
    adapter: Optional[OmniLindyK8sAdapter] = None,
    engine: str = "omni_mrl",
) -> K8sExtenderServer:
    """`K8sExtenderServer` 인스턴스를 생성해 반환한다 (테스트 시 `port=0` 사용 가능)."""
    adp = OmniLindyK8sAdapter(nodes_per_rack=nodes_per_rack, engine=engine) if adapter is None else adapter
    return K8sExtenderServer((host, port), _ExtenderRequestHandler, adapter=adp, quiet=quiet)


def main(argv: Optional[list[str]] = None) -> None:
    """K8s Scheduler Extender HTTP 서버 CLI 진입점."""
    parser = argparse.ArgumentParser(description="Omni-Lindy / Omni-MRL Kubernetes Scheduler HTTP Extender (v11.1)")
    parser.add_argument("--host", default="127.0.0.1", help="바인딩 호스트 주소 (기본: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8888, help="수신 HTTP 포트 (기본: 8888)")
    parser.add_argument("--nodes-per-rack", type=int, default=8, help="기본 랙당 노드 수 (기본: 8)")
    parser.add_argument(
        "--engine",
        choices=("omni_mrl", "omni_lindy"),
        default="omni_mrl",
        help="스케줄링 코어 엔진 (기본: omni_mrl v11.1)",
    )
    parser.add_argument("--quiet", action="store_true", help="요청 액세스 로그 비활성화")
    args = parser.parse_args(argv)

    server = create_extender_server(
        host=args.host,
        port=args.port,
        nodes_per_rack=args.nodes_per_rack,
        quiet=args.quiet,
        engine=args.engine,
    )
    actual_host, actual_port = server.server_address[:2]
    print(
        f"[Omni-Lindy/Omni-MRL Extender] http://{actual_host}:{actual_port} 에서 대기 중 "
        f"(engine={args.engine}, nodes_per_rack={args.nodes_per_rack})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Omni-Lindy Extender] 종료 신호 수신, 서버를 정상 종료합니다.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
