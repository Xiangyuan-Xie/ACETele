"""Exact-peer CURVE certificate loading and ZAP authorization."""

from __future__ import annotations

from pathlib import Path

import zmq
from ace_robot_zmq.options import CurveCredentials
from zmq.auth import load_certificate
from zmq.auth.thread import ThreadAuthenticator


class _ExactPeerCredentials:
    """Authorize only the configured peer's Z85-encoded public key."""

    def __init__(self, public_key: bytes) -> None:
        self._public_key = public_key

    def callback(self, _domain: str, key: bytes) -> bool:
        return key == self._public_key


class CurveAuthenticator:
    """Own the ZAP thread shared by the binding and connecting sockets."""

    def __init__(self, context: zmq.Context, credentials: CurveCredentials) -> None:
        local_public, local_secret = load_certificate(credentials.secret_key)
        peer_public, _ = load_certificate(credentials.peer_key)
        if local_secret is None:
            raise ValueError("CURVE local certificate must contain a secret key")
        self.local_public = local_public
        self.local_secret = local_secret
        self.peer_public = peer_public
        self._authenticator = ThreadAuthenticator(context)
        self._authenticator.start()
        try:
            self._authenticator.configure_curve_callback(
                domain="*",
                credentials_provider=_ExactPeerCredentials(peer_public),
            )
        except BaseException:
            self._authenticator.stop()
            raise

    def configure_server(self, socket: zmq.Socket) -> None:
        socket.curve_publickey = self.local_public
        socket.curve_secretkey = self.local_secret
        socket.curve_server = True

    def configure_client(self, socket: zmq.Socket) -> None:
        socket.curve_publickey = self.local_public
        socket.curve_secretkey = self.local_secret
        socket.curve_serverkey = self.peer_public

    def close(self) -> None:
        self._authenticator.stop()


def generate_curve_certificates(directory: Path, name: str) -> tuple[Path, Path]:
    """Create one public/private certificate pair with private-file permissions."""

    if not isinstance(name, str) or not name.strip() or Path(name).name != name:
        raise ValueError("certificate name must be one non-empty path component")
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    public_path, secret_path = zmq.auth.create_certificates(target, name)
    Path(secret_path).chmod(0o600)
    return Path(public_path), Path(secret_path)


__all__ = ["CurveAuthenticator", "generate_curve_certificates"]
