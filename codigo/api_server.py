"""
Servidor da API REST do dataset de indicadores CTI.

Uso:
    python api_server.py
    python api_server.py --host 0.0.0.0 --port 8000
    python api_server.py --reload          # modo dev com hot-reload
"""
import argparse
import logging

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CTI Dataset API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (padrão: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Porta (padrão: 8000)")
    parser.add_argument("--reload", action="store_true", help="Hot-reload para desenvolvimento")
    args = parser.parse_args()

    print(f"\n  CTI Dataset API → http://localhost:{args.port}")
    print(f"  Documentação    → http://localhost:{args.port}/docs\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
